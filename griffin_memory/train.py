"""Single-device scratch pretraining with TBPTT, AMP, evaluation and full resume."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .data import Cursor, DocumentDataset, DocumentStream, dataset_manifest
from .model import GriffinMemoryLM, StreamState
from .tokenizer import Tokenizer


DEFAULTS = dict(batch_size=2, accumulation_steps=1, max_steps=1000, epochs=1,
                learning_rate=3e-4, min_lr_ratio=0.1, warmup_steps=100,
                weight_decay=0.1, clip_grad=1.0, seed=42, precision="fp32", deterministic=False)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    return device


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def learning_rate(step, options):
    warmup = options["warmup_steps"]
    if step < warmup:
        return options["learning_rate"] * (step + 1) / warmup
    # step is zero-based; the final planned update reaches min_lr_ratio.
    progress = min(1.0, max(0.0, (step - warmup) / max(1, options["max_steps"] - warmup - 1)))
    ratio = options["min_lr_ratio"] + (1 - options["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))
    return options["learning_rate"] * ratio


def atomic_save(payload, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    # Checkpoints use only tensors and primitive containers, not pickled model objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def forward_batch(model, batch, state, device):
    if batch["reset"]:
        state = None
    targets = batch["targets"].to(device)
    logits, state = model(batch["tokens"].to(device), state, targets != -100,
                          batch["document_ids"], batch["metadata"])
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss, int((targets != -100).sum()), state.detach()


@torch.no_grad()
def evaluate(model, dataset, options, device, max_batches):
    if not len(dataset):
        return None
    was_training = model.training
    model.eval()
    stream = DocumentStream(dataset, options["batch_size"], model.config.chunk_size,
                            options["seed"], shuffle=False)
    total_loss, tokens, batches, state = 0.0, 0, 0, None
    try:
        while max_batches == 0 or batches < max_batches:
            batch = stream.next()
            if batch is None:
                break
            with amp_context(device, options["precision"]):
                loss, count, state = forward_batch(model, batch, state, device)
            total_loss += float(loss)
            tokens += count
            batches += 1
    finally:
        model.train(was_training)
    mean = total_loss / tokens
    return {"val_loss": mean, "val_perplexity": math.exp(mean) if mean < 700 else float("inf"),
            "val_tokens": tokens, "val_batches": batches}


def train(args):
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        raise ValueError("This reference trainer is single-device. Do not launch with multi-rank torchrun.")
    resumed = load_checkpoint(args.resume) if args.resume else None
    options = {}
    for key, default in DEFAULTS.items():
        requested = getattr(args, key)
        saved = resumed["options"][key] if resumed else default
        if resumed and requested is not None and requested != saved:
            raise ValueError(f"Resume mismatch for {key}: checkpoint={saved}, requested={requested}")
        options[key] = saved if requested is None else requested
    for key in ("batch_size", "accumulation_steps", "max_steps", "epochs"):
        if options[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if options["learning_rate"] <= 0 or options["clip_grad"] <= 0 or options["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= options["min_lr_ratio"] <= 1 or options["warmup_steps"] < 0:
        raise ValueError("Invalid scheduler settings")
    if min(args.eval_every, args.save_every, args.eval_batches) < 0 or args.log_every <= 0:
        raise ValueError("Intervals must be nonnegative; log-every must be positive")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = choose_device(args.device)
    if options["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")
    if options["precision"] == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; use fp16 or fp32")
    if args.threads:
        torch.set_num_threads(args.threads)
    random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    torch.use_deterministic_algorithms(options["deterministic"])
    torch.backends.cudnn.benchmark = False
    manifest, fingerprint = dataset_manifest(args.data)
    tokenizer = Tokenizer.load(Path(args.data) / "tokenizer.json")
    if tokenizer.vocab_size != manifest["vocab_size"]:
        raise ValueError("Tokenizer vocabulary disagrees with dataset")
    if resumed:
        if fingerprint != resumed["data_fingerprint"]:
            raise ValueError("Cannot resume on a different dataset/tokenizer")
        config = ModelConfig(**resumed["config"])
        if args.config:
            requested = ModelConfig.load(args.config)
            requested.vocab_size = tokenizer.vocab_size
            if requested.to_dict() != config.to_dict():
                raise ValueError("Config does not match checkpoint")
    else:
        if not args.config:
            raise ValueError("--config is required for a new run")
        config = ModelConfig.load(args.config)
        config.vocab_size = tokenizer.vocab_size
    training_data, validation_data = DocumentDataset(args.data, "train"), DocumentDataset(args.data, "val")
    if not len(training_data):
        raise ValueError("No training documents")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not resumed and any(out.iterdir()):
        raise ValueError("Output directory must be empty for a new run; use --resume or another --out")
    model = GriffinMemoryLM(config).to(device)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": options["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=options["learning_rate"], betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=options["precision"] == "fp16")
    step, total_tokens, best_loss, state, cursor = 0, 0, float("inf"), None, Cursor()
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        step, total_tokens, best_loss = resumed["step"], resumed["total_tokens"], resumed["best_loss"]
        cursor = Cursor(**resumed["cursor"])
        if resumed["stream_state"] is not None:
            state = StreamState.from_state_dict(config, resumed["stream_state"], device)
        torch.set_rng_state(resumed["rng_cpu"])
        random.setstate(resumed["rng_python"])
        if device.type == "cuda" and resumed["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(resumed["rng_cuda"])
    stream = DocumentStream(training_data, options["batch_size"], config.chunk_size,
                            options["seed"], options["epochs"], cursor=cursor)
    tokenizer.save(out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (out / "run.json").write_text(json.dumps({"options": options, "data_fingerprint": fingerprint,
                                            "torch": str(torch.__version__), "device": str(device)}, indent=2) + "\n")
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                      "device": str(device), "precision": options["precision"], "resume_step": step}))

    def save(name):
        atomic_save({"format_version": 1, "config": config.to_dict(), "options": options,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "step": step, "total_tokens": total_tokens, "best_loss": best_loss,
                     "cursor": stream.cursor.state_dict(), "data_fingerprint": fingerprint,
                     "tokenizer_json": (out / "tokenizer.json").read_text(),
                     "stream_state": state.state_dict() if state is not None and stream.cursor.chunk else None,
                     "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                     "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, out / name)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated, token_count, loss_sum = 0, 0, 0.0
    start_time, start_tokens = time.monotonic(), total_tokens
    last_evaluated = None
    last_metrics = None
    with (out / "train.jsonl").open("a") as log:
        def emit(record):
            line = json.dumps(record)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        while step < options["max_steps"] and (args.stop_after_steps is None or step < args.stop_after_steps):
            batch = stream.next()
            if batch is not None:
                with amp_context(device, options["precision"]):
                    loss, count, state = forward_batch(model, batch, state, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; previous checkpoint remains valid")
                scaler.scale(loss).backward()
                accumulated += 1
                token_count += count
                loss_sum += float(loss.detach())
            if accumulated and (accumulated == options["accumulation_steps"] or batch is None):
                scaler.unscale_(optimizer)
                # Token-weighted accumulation, including uneven/padded final chunks.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(token_count)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["clip_grad"],
                                                     error_if_nonfinite=options["precision"] != "fp16")
                lr = learning_rate(step, options)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                optimizer.zero_grad(set_to_none=True)
                step += 1  # attempted optimizer updates, including AMP-overflow skips
                total_tokens += token_count
                if step % args.log_every == 0 or step == 1:
                    emit({"step": step, "train_loss": loss_sum / token_count, "lr": lr,
                          "grad_norm": float(norm), "amp_skipped": skipped, "tokens": total_tokens,
                          "tokens_per_second": (total_tokens - start_tokens) / max(1e-6, time.monotonic() - start_time),
                          "cursor": stream.cursor.state_dict(),
                          "memory_entries": sum(len(bank.entries) for bank in state.banks)})
                accumulated, token_count, loss_sum = 0, 0, 0.0
                if args.eval_every and step % args.eval_every == 0:
                    last_metrics = evaluate(model, validation_data, options, device, args.eval_batches)
                    last_evaluated = step
                    if last_metrics:
                        emit({"step": step, **last_metrics})
                        if last_metrics["val_loss"] < best_loss:
                            best_loss = last_metrics["val_loss"]
                            save("best.pt")
                if args.save_every and step % args.save_every == 0:
                    save("last.pt")
            if batch is None:
                break
        if last_evaluated != step:
            last_metrics = evaluate(model, validation_data, options, device, args.eval_batches)
            if last_metrics:
                emit({"step": step, **last_metrics})
                if last_metrics["val_loss"] < best_loss:
                    best_loss = last_metrics["val_loss"]
                    save("best.pt")
        save("last.pt")
        emit({"event": "finished", "step": step, "tokens": total_tokens,
              "checkpoint": str(out / "last.pt"), "epoch": stream.cursor.epoch})
    return {"step": step, "tokens": total_tokens, "validation": last_metrics}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--resume")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0, help="CPU intra-op threads; 0 preserves PyTorch default")
    for name in ("batch_size", "accumulation_steps", "max_steps", "epochs", "warmup_steps", "seed"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name in ("learning_rate", "min_lr_ratio", "weight_decay", "clip_grad"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--eval-every", type=int, default=100, help="0 disables periodic evaluation (final eval still runs)")
    p.add_argument("--save-every", type=int, default=100, help="0 disables periodic saves (final save still runs)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=32, help="0 evaluates entire validation corpus")
    p.add_argument("--stop-after-steps", type=int, help="Stop cleanly at this update without changing LR schedule")
    return p


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
