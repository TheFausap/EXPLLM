"""Train the QALF proof-of-concept."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from qalf.data import build_tokenizer, encode_examples, make_windows, read_jsonl, relation_counts, trigram_counts
from qalf.joblog import JobLogger
from qalf.model import QALFConfig, QALFModel, cross_entropy_with_l2, device_for_training, load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QALF")
    parser.add_argument("--data", default="data/seed_corpus.jsonl")
    parser.add_argument("--out", default="runs/qalf_poc")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=24)
    parser.add_argument("--relations", type=int, default=4)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--attractor-limit", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--trigram-top-k", type=int, default=32)
    parser.add_argument("--trigram-min-count", type=int, default=2)
    parser.add_argument("--trigram-strength", type=float, default=0.75)
    parser.add_argument("--bigram-strength", type=float, default=0.35)
    parser.add_argument("--bigram-decay-epochs", type=int, default=0, help="Linearly decay bigram strength to 0 over this many epochs from the first training epoch. 0 = constant.")
    parser.add_argument("--trigram-decay-epochs", type=int, default=0, help="Linearly decay trigram strength to 0 over this many epochs from the first training epoch. 0 = constant.")
    parser.add_argument("--resume", default=None, help="Resume from a QALF checkpoint with training_state")
    parser.add_argument("--reset-optimizer", action="store_true", help="Discard the saved optimizer state on resume (needed when model architecture changed)")
    parser.add_argument("--reset-lr-schedule", action="store_true", help="Reset the LR schedule position to 0 on resume, so warmup fires again")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint_epoch_N.pt every N epochs")
    parser.add_argument("--lr-schedule", choices=["constant", "warmup-cosine"], default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--bigram-device", default=None, help="Device to store the bigram buffer, e.g. 'cuda:1' or 'cpu'. Frees ~1 GB on the training GPU.")
    parser.add_argument("--entropy-weight", type=float, default=0.0, help="Coefficient for mixture-entropy regularisation (e.g. 0.1). Resists component collapse.")
    return parser.parse_args()


def scheduled_lr(
    base_lr: float,
    global_step: int,
    total_steps: int,
    schedule: str,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> float:
    if schedule == "constant" or total_steps <= 1:
        return base_lr
    warmup_steps = max(1, int(total_steps * max(warmup_fraction, 0.0)))
    min_lr = base_lr * min_lr_ratio
    if global_step < warmup_steps:
        return base_lr * float(global_step + 1) / float(warmup_steps)
    progress = min(1.0, (global_step - warmup_steps) / max(total_steps - warmup_steps, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


_BUFFER_KEYS = frozenset({"bigram_logits", "trigram_keys", "trigram_token_ids", "trigram_token_logits"})


def _restore_exp_avg_sq(
    optimizer: torch.optim.Adam,
    saved_opt_state: dict,
    saved_model_state: dict,
    model: "QALFModel",
) -> int:
    """Partial optimizer restore for architecture-mismatched checkpoints.

    Copies exp_avg_sq (gradient scale) from the saved state to the current optimizer
    for parameters that exist in both versions, matched by name. exp_avg is zeroed
    to clear stale direction bias. New parameters start with fresh state.
    Returns the number of parameters restored.
    """
    saved_param_names = [k for k in saved_model_state if k not in _BUFFER_KEYS]
    saved_states = saved_opt_state.get("state", {})
    name_to_saved = {name: saved_states[i] for i, name in enumerate(saved_param_names) if i in saved_states}
    restored = 0
    for name, param in model.named_parameters():
        s = name_to_saved.get(name)
        if s is None or "exp_avg_sq" not in s:
            continue
        step = s.get("step", torch.tensor(0.0))
        optimizer.state[param] = {
            "step": step.clone() if isinstance(step, torch.Tensor) else torch.tensor(float(step)),
            "exp_avg": torch.zeros_like(param.data),
            "exp_avg_sq": s["exp_avg_sq"].clone().to(param.device),
        }
        restored += 1
    return restored


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = JobLogger(args.log_file)
    examples = read_jsonl(args.data)
    logger.emit({"stage": "read_data", "examples": len(examples), "data": args.data})
    resumed_payload = None
    resumed_metadata = {}
    resumed_training_state = {}
    if args.resume:
        resumed_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model, tokenizer, resumed_metadata = load_checkpoint(args.resume, map_location="cpu")
        resumed_training_state = resumed_payload.get("training_state", {})
        config = model.config
        # CLI args override checkpoint config for values that don't affect weight shapes
        config.bigram_strength = args.bigram_strength
        config.trigram_strength = args.trigram_strength
        last_lr = None
        if resumed_training_state.get("optimizer_state"):
            groups = resumed_training_state["optimizer_state"].get("param_groups", [])
            if groups:
                last_lr = groups[0].get("lr")
        logger.emit({"stage": "resume_checkpoint", "checkpoint": args.resume, "start_epoch": int(resumed_training_state.get("epoch", 0)) + 1, "vocab": len(tokenizer.vocab), "checkpoint_last_lr": last_lr})
    else:
        tokenizer = build_tokenizer(examples, vocab_size=args.vocab_size)
        config = QALFConfig(
            vocab_size=len(tokenizer.vocab),
            dimension=args.dimension,
            context_size=args.context_size,
            num_relations=args.relations,
            num_components=args.components,
            bigram_strength=args.bigram_strength,
            trigram_strength=args.trigram_strength,
            pad_id=tokenizer.pad_id,
        )
        model = None
        logger.emit({"stage": "build_tokenizer", "vocab": len(tokenizer.vocab)})
    encoded = encode_examples(tokenizer, examples)
    contexts, prev2_tokens, prev_tokens, targets = make_windows(encoded, config.context_size, tokenizer.pad_id, include_prev2=True)
    if args.max_windows is not None and args.max_windows < targets.numel():
        order = torch.randperm(targets.numel())[: args.max_windows]
        contexts = contexts[order]
        prev2_tokens = prev2_tokens[order]
        prev_tokens = prev_tokens[order]
        targets = targets[order]
    window_ram_gb = (contexts.numel() + prev2_tokens.numel() + prev_tokens.numel() + targets.numel()) * contexts.element_size() / (1024 ** 3)
    logger.emit({"stage": "make_windows", "windows": int(targets.numel()), "context_size": config.context_size, "window_tensor_gb": window_ram_gb})
    bigram = relation_counts(encoded, len(tokenizer.vocab))
    trigram = trigram_counts(encoded, len(tokenizer.vocab), top_k=args.trigram_top_k, min_count=args.trigram_min_count)
    bigram_gb = bigram.numel() * bigram.element_size() / (1024 ** 3)
    trigram_gb = sum(t.numel() * t.element_size() for t in trigram.values()) / (1024 ** 3)
    logger.emit({"stage": "build_higher_order_memory", "trigram_contexts": int(trigram["keys"].numel()), "trigram_top_k": args.trigram_top_k, "bigram_gb": bigram_gb, "trigram_gb": trigram_gb})
    device = device_for_training(args.device)
    bigram_device = torch.device(args.bigram_device) if args.bigram_device else device
    logger.emit({"stage": "init_model", "device": str(device), "bigram_device": str(bigram_device), "dimension": config.dimension, "relations": config.num_relations, "components": config.num_components})
    if model is None:
        model = QALFModel(config, bigram_logits=bigram, trigram_prior=trigram)
    else:
        model.bigram_logits = bigram
        model.trigram_keys = trigram["keys"]
        model.trigram_token_ids = trigram["token_ids"]
        model.trigram_token_logits = trigram["token_logits"]
    model = model.to(device)
    if bigram_device != device:
        model.bigram_logits = model.bigram_logits.to(bigram_device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = TensorDataset(contexts, prev2_tokens, prev_tokens, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if resumed_training_state.get("optimizer_state"):
        if not args.reset_optimizer:
            try:
                optimizer.load_state_dict(resumed_training_state["optimizer_state"])
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
            except ValueError:
                n = _restore_exp_avg_sq(optimizer, resumed_training_state["optimizer_state"], resumed_payload["model_state"], model)
                logger.emit({"stage": "partial_optimizer_restore", "reason": "architecture_mismatch", "parameters_restored": n})
        else:
            n = _restore_exp_avg_sq(optimizer, resumed_training_state["optimizer_state"], resumed_payload["model_state"], model)
            logger.emit({"stage": "partial_optimizer_restore", "reason": "reset_optimizer", "parameters_restored": n})
    start_epoch = int(resumed_training_state.get("epoch", 0)) + 1
    steps_per_epoch = len(loader)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    global_step = int(resumed_training_state.get("global_step", max(start_epoch - 1, 0) * steps_per_epoch))
    if args.reset_lr_schedule and args.resume:
        # Calibrate the schedule to the remaining epochs so warmup/cosine covers
        # actual training time rather than the full epoch range from epoch 0.
        remaining_epochs = max(args.epochs - start_epoch + 1, 1)
        total_steps = max(steps_per_epoch * remaining_epochs, 1)
    lr_step = 0 if args.reset_lr_schedule else global_step
    logger.emit({"stage": "lr_schedule", "schedule": args.lr_schedule, "base_lr": args.lr, "warmup_fraction": args.warmup_fraction, "min_lr_ratio": args.min_lr_ratio, "start_global_step": global_step, "start_lr_step": lr_step, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch})
    history: list[dict[str, float]] = list(resumed_metadata.get("history", []))
    if args.resume:
        model.eval()
        pre_loss, pre_seen = 0.0, 0
        with torch.no_grad():
            for i, (bc, bp2, bp, bt) in enumerate(loader):
                if i >= 20:
                    break
                logits = model(bc.to(device), bp.to(device), bp2.to(device))
                pre_loss += float(F.cross_entropy(logits, bt.to(device)).cpu()) * bt.numel()
                pre_seen += bt.numel()
        logger.emit({"stage": "resume_eval", "ce_loss": pre_loss / max(pre_seen, 1), "batches": min(20, len(loader))})
        model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.bigram_decay_epochs > 0:
            t = max(0.0, min(1.0, (epoch - start_epoch) / args.bigram_decay_epochs))
            model.config.bigram_strength = args.bigram_strength * (1.0 - t)
        if args.trigram_decay_epochs > 0:
            t = max(0.0, min(1.0, (epoch - start_epoch) / args.trigram_decay_epochs))
            model.config.trigram_strength = args.trigram_strength * (1.0 - t)
        total_loss = 0.0
        seen = 0
        for batch_contexts, batch_prev2, batch_prev, batch_targets in loader:
            batch_contexts = batch_contexts.to(device)
            batch_prev2 = batch_prev2.to(device)
            batch_prev = batch_prev.to(device)
            batch_targets = batch_targets.to(device)
            lr_now = scheduled_lr(args.lr, lr_step, total_steps, args.lr_schedule, args.warmup_fraction, args.min_lr_ratio)
            set_optimizer_lr(optimizer, lr_now)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_contexts, batch_prev, batch_prev2)
            loss = cross_entropy_with_l2(model, logits, batch_targets, entropy_weight=args.entropy_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            lr_step += 1
            total_loss += float(loss.detach().cpu()) * batch_targets.numel()
            seen += batch_targets.numel()
        avg_loss = total_loss / max(seen, 1)
        if epoch == 1 or epoch == start_epoch or epoch % args.log_every == 0 or epoch == args.epochs:
            diag_contexts = contexts[: min(16, len(contexts))].to(device)
            diag = model.diagnostics(diag_contexts)
            record = {"epoch": epoch, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"], "global_step": global_step, **diag}
            history.append(record)
            logger.emit(record)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                out_dir / f"checkpoint_epoch_{epoch}.pt",
                model.cpu(),
                tokenizer,
                metadata={"data": args.data, "device": str(device), "epochs": args.epochs, "examples": len(examples), "windows": int(targets.numel()), "history": history},
                training_state={"epoch": epoch, "global_step": global_step, "optimizer_state": optimizer.state_dict()},
            )
            model.to(device)
            if bigram_device != device:
                model.bigram_logits = model.bigram_logits.to(bigram_device)
            logger.emit({"stage": "checkpoint", "epoch": epoch, "checkpoint": str(out_dir / f"checkpoint_epoch_{epoch}.pt")})
    metadata = {
        "data": args.data,
        "device": str(device),
        "epochs": args.epochs,
        "examples": len(examples),
        "seed_examples": [
            {"prompt": example.prompt, "reply": example.reply}
            for example in examples[: args.attractor_limit]
        ],
        "attractor_limit": args.attractor_limit,
        "windows": int(targets.numel()),
        "history": history,
    }
    save_checkpoint(out_dir / "model.pt", model.cpu(), tokenizer, metadata, training_state={"epoch": args.epochs, "global_step": global_step, "optimizer_state": optimizer.state_dict()})
    logger.emit({"stage": "saved", "checkpoint": str(out_dir / "model.pt")})
    logger.close()


if __name__ == "__main__":
    main()
