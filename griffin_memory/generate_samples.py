"""Batch held-out generation: continuations, reference text and memory diagnostics.

Unlike `generate.py` (one interactive prompt), this module is built for run
reports: it takes held-out documents from a prepared corpus, streams an optional
long prefill, generates continuations, scores the *real* continuation tokens with
next-token cross-entropy (retrieval tier on and off), and writes Markdown + JSON.

Prompts come from the requested split's own tokens, so generation uses exactly
the same tokenizer and document framing as training.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .data import DocumentDataset, dataset_manifest
from .generate import sample
from .model import GriffinMemoryLM
from .tokenizer import Tokenizer, BOS, EOS
from .train import amp_context, choose_device, load_checkpoint


@contextmanager
def retrieval_disabled(model):
    """Ablation: keep recurrent and attention state, but make the ANN tier add zeros."""
    original = model.memory.forward
    model.memory.forward = lambda hidden, banks: torch.zeros_like(hidden)
    try:
        yield
    finally:
        model.memory.forward = original


def load_model(checkpoint_path, device):
    """Load weights and the checkpoint-embedded tokenizer; no external artifacts."""
    if not Path(checkpoint_path).is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    config = ModelConfig(**checkpoint["config"])
    model = GriffinMemoryLM(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    with tempfile.TemporaryDirectory() as temp:
        tokenizer_path = Path(temp) / "tokenizer.json"
        tokenizer_path.write_text(checkpoint["tokenizer_json"])
        tokenizer = Tokenizer.load(tokenizer_path)
    return model, tokenizer, checkpoint


def select_documents(dataset, count, order, seed, offset=0, ids=None):
    """Deterministic document selection, including explicit ids and length order."""
    records = dataset.metadata
    if ids:
        wanted = {value.strip() for value in ids.split(",") if value.strip()}
        found = [index for index, record in enumerate(records) if str(record.get("id")) in wanted]
        missing = wanted - {str(records[index].get("id")) for index in found}
        if missing:
            raise ValueError(f"Document id(s) not found in this split: {sorted(missing)}")
        return found
    lengths = [len(dataset[index]) for index in range(len(records))]
    if order == "longest":
        pool = sorted(range(len(records)), key=lambda index: (-lengths[index], index))
    elif order == "shortest":
        pool = sorted(range(len(records)), key=lambda index: (lengths[index], index))
    else:
        pool = list(range(len(records)))
    pool = pool[offset:]
    if order == "random":
        random.Random(seed).shuffle(pool)
    return pool[:count]


def window(document, prompt_tokens, prefill_tokens, start_fraction):
    """Locate the streamed window of one document.

    Returns (streamed ids starting with BOS, context length in streamed positions
    including BOS, exclusive index into the document's text tokens where the
    continuation starts). The context length is the scoring boundary: the first
    continuation token is predicted at that streamed position.
    """
    text = document[1:]  # document[0] is BOS
    span = prompt_tokens + prefill_tokens
    if span <= 0:
        raise ValueError("prompt-tokens + prefill-tokens must be positive")
    start = int(start_fraction * max(0, len(text) - span))
    end = start + span
    if end > len(text):
        raise ValueError(f"Document has {len(text)} text tokens but {end} were requested; "
                         "lower --prompt-tokens/--prefill-tokens/--start-fraction")
    streamed = [BOS] + [int(token) for token in text[start:end]]
    return streamed, len(streamed), end


def stream_tokens(model, ids, device, state=None, document_id="sample:0", score_from=None):
    """Stream ids without crossing a memory chunk boundary inside one call.

    Returns (logits for the final token, state, summed CE, scored token count).
    `score_from` is the first absolute index of `ids` whose next-token loss counts.
    """
    logits, loss_sum, loss_count, offset = None, 0.0, 0, 0
    while offset < len(ids):
        position = state.position if state is not None else 0
        room = model.config.chunk_size - position % model.config.chunk_size
        count = min(room, len(ids) - offset)
        if count <= 0:
            raise ValueError("Internal window error: empty stream step")
        chunk = torch.tensor([ids[offset:offset + count]], dtype=torch.long, device=device)
        logits, state = model(chunk, state, document_ids=[document_id])
        if score_from is not None:
            # logits[j] predicts ids[offset + j + 1]; the final token has no target yet.
            targets = ids[offset + 1:offset + count + 1]
            if targets:
                chunk_logits = logits[0, :len(targets)].float()
                # Target j of this chunk is the token at absolute index offset + j + 1.
                positions = torch.arange(offset + 1, offset + 1 + len(targets), device=device)
                mask = positions >= score_from
                if bool(mask.any()):
                    loss_sum += float(F.cross_entropy(chunk_logits[mask],
                                                      torch.tensor(targets, device=device)[mask],
                                                      reduction="sum"))
                    loss_count += int(mask.sum())
        offset += count
    return logits[:, -1], state, loss_sum, loss_count


def decode_from_prompt(model, ids, device, decoding, max_new_tokens, memory=True, document_id="sample:0"):
    """Prefill the window (long-prompt safe) then decode token by token."""
    started = time.monotonic()
    torch.manual_seed(decoding["seed"])
    with (nullcontext() if memory else retrieval_disabled(model)), torch.inference_mode():
        logits, state, _, _ = stream_tokens(model, ids, device, document_id=document_id)
        pieces = []
        for _ in range(max_new_tokens):
            token = int(sample(logits, decoding["temperature"], decoding["top_k"], decoding["top_p"]).item())
            if token == EOS:
                break
            pieces.append(token)
            logits, state = model(torch.tensor([[token]], dtype=torch.long, device=device), state,
                                  document_ids=[document_id])
            logits = logits[:, -1]
        diagnostics = {"memory_entries": [len(bank.entries) for bank in state.banks],
                       "pending_tokens": len(state.pending_tokens[0]), "position": int(state.position)}
    return {"tokens": pieces, "seconds": round(time.monotonic() - started, 3), **diagnostics}


def score_continuation(model, ids, prompt_length, device, memory=True, document_id="sample:0"):
    """Teacher-forced CE of the real next tokens, with writes/reads as configured."""
    with (nullcontext() if memory else retrieval_disabled(model)), torch.inference_mode():
        _, state, loss_sum, loss_count = stream_tokens(model, ids, device, document_id=document_id,
                                                       score_from=prompt_length)
        entries = [len(bank.entries) for bank in state.banks]
    if not loss_count:
        raise ValueError("No continuation tokens were scored; the document ends too early")
    mean = loss_sum / loss_count
    return {"loss": mean, "tokens": loss_count,
            "perplexity": float("inf") if mean > 700 else float(torch.exp(torch.tensor(mean))),
            "memory_entries": entries}


def clip_middle(text, limit):
    if limit <= 0 or len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters omitted] …\n{text[-half:]}"


def render_markdown(report):
    decoding = report["decoding"]
    lines = [f"# Generation report — checkpoint step {report['checkpoint_step']}",
             "",
             f"- checkpoint: `{report['checkpoint']}`",
             f"- corpus: `{report['data']}` (split `{report['split']}`), tokenizer `{report['tokenizer']['type']}` "
             f"vocab {report['tokenizer']['vocab_size']}",
             f"- decoding: greedy and temperature {decoding['temperature']}, top-k {decoding['top_k']}, "
             f"top-p {decoding['top_p']}, seed {decoding['seed']}",
             f"- window: {report['context_tokens']} streamed positions before generation "
             f"(1 BOS + {report['prefill_tokens']} prefill + {report['prompt_window']} prompt)"
             + (f", window starts at fraction {report['start_fraction']} of the document" if report["start_fraction"] else "")
             + f"; up to {report['max_new_tokens']} new tokens; {report['continuation_tokens']} reference tokens",
             "",
             "`CE` is next-token cross-entropy on the real held-out continuation, teacher-forced. `mem off` "
             "disables only the retrieval tier (recurrent and attention state stay active). Neither number "
             "measures narrative quality — read the text below.",
             "",
             "| # | document | streamed | new (greedy/sampled) | CE mem on | CE mem off | perplexity | chunks |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for index, item in enumerate(report["samples"], start=1):
        scores, greedy = item["scores"], item["continuations"]["greedy"]
        off = f"{scores['memory_off']['loss']:.3f}" if scores["memory_off"] else "not run"
        lines.append(f"| {index} | `{item['document_id']}` | {item['context_tokens']} | "
                     f"{len(greedy['tokens'])}/{len(item['continuations']['sampled']['tokens'])} | "
                     f"{scores['memory_on']['loss']:.3f} | {off} | {scores['memory_on']['perplexity']:.1f} | "
                     f"{greedy['memory_entries'][0]} |")
    aggregate = report["aggregate"]
    lines += ["", f"Aggregate continuation CE: **{aggregate['continuation_loss']:.4f}** over "
                  f"{aggregate['continuation_tokens']} held-out tokens (perplexity "
                  f"{aggregate['continuation_perplexity']:.2f})"
              + (f", retrieval-off CE {aggregate['memory_off_loss']:.4f}" if aggregate.get("memory_off_loss") else "")]
    for index, item in enumerate(report["samples"], start=1):
        greedy, sampled = item["continuations"]["greedy"], item["continuations"]["sampled"]
        lines += ["", "---", "",
                  f"## Sample {index} — `{item['document_id']}`",
                  "",
                  f"- document: `{item.get('title') or 'untitled'}`; {item['memory']['document_tokens']} tokens in "
                  f"this split; {item['context_tokens']} streamed positions before generation",
                  f"- seconds: prefill {item['seconds']['prefill']}, greedy {greedy['seconds']}, "
                  f"sampled {sampled['seconds']}",
                  f"- retrieval chunks when decoding: {greedy['memory_entries']} (capacity "
                  f"{item['memory']['capacity']}), pending tokens {greedy['pending_tokens']}",
                  f"- continuation CE: mem on {item['scores']['memory_on']['loss']:.4f} "
                  f"(ppl {item['scores']['memory_on']['perplexity']:.2f}, {item['scores']['memory_on']['tokens']} tokens)"
                  + (f", mem off {item['scores']['memory_off']['loss']:.4f} "
                     f"(ppl {item['scores']['memory_off']['perplexity']:.2f})" if item["scores"]["memory_off"] else ""),
                  "",
                  "### Prompt (real held-out text)", "", "```text", item["prompt_text"], "```", "",
                  "### Greedy continuation", "", "```text", greedy["text"] or "[immediate EOS]", "```", "",
                  f"### Sampled continuation (temperature {decoding['temperature']}, top-k {decoding['top_k']}, "
                  f"top-p {decoding['top_p']})" + (" — identical to greedy (temperature 0 or --decoding greedy)"
                                                   if sampled.get("same_as_greedy") else ""),
                  "", "```text", sampled["text"] or "[immediate EOS]", "```", "",
                  "### Real held-out continuation (reference)", "", "```text", item["reference_text"], "```"]
    return "\n".join(lines) + "\n"


def generate_samples(args):
    device = choose_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires CUDA")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 is not supported on this GPU")
    if min(args.documents, args.prompt_tokens, args.max_new_tokens, args.continuation_tokens, args.markdown_characters) < 0 \
            or args.documents == 0:
        raise ValueError("Document and token counts must be nonnegative; documents must be positive")
    if args.prefill_tokens < 0 or args.document_offset < 0 or not 0 <= args.start_fraction < 1:
        raise ValueError("prefill-tokens and document-offset must be nonnegative; start-fraction is in [0, 1)")
    if args.temperature < 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid decoding settings")
    if args.threads:
        torch.set_num_threads(args.threads)
    manifest, _ = dataset_manifest(args.data)
    if not (Path(args.data) / f"{args.split}.metadata.jsonl").is_file():
        raise ValueError(f"The dataset does not include split {args.split!r}")
    dataset = DocumentDataset(args.data, args.split)
    if not len(dataset):
        raise ValueError("The selected split has no documents")
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("Checkpoint tokenizer and model vocabulary disagree")
    indices = select_documents(dataset, args.documents, args.select, args.seed,
                               offset=args.document_offset, ids=args.document_ids)
    if not indices:
        raise ValueError("Document selection is empty; check --documents/--document-offset/--document-ids")
    base_decoding = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed}
    greedy_decoding = {**base_decoding, "temperature": 0.0}
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint["step"], "data": str(args.data),
              "split": args.split, "dataset_stats": manifest["stats"].get(args.split, {}),
              "tokenizer": {"type": "bpe" if tokenizer.backend is not None else "utf8_bytes",
                            "vocab_size": tokenizer.vocab_size},
              "model_config": model.config.to_dict(), "decoding": base_decoding,
              "decoding_requested": args.decoding, "prompt_window": args.prompt_tokens,
              "context_tokens": 1 + args.prompt_tokens + args.prefill_tokens,
              "max_new_tokens": args.max_new_tokens, "continuation_tokens": args.continuation_tokens,
              "prefill_tokens": args.prefill_tokens, "start_fraction": args.start_fraction,
              "retrieval_ablation": not args.no_ablation, "torch": str(torch.__version__),
              "device": str(device), "precision": args.precision, "samples": []}
    total_loss, total_tokens, total_off_loss = 0.0, 0, 0.0
    with torch.inference_mode(), amp_context(device, args.precision):
        for index in indices:
            document = dataset[index]
            ids, prompt_length, text_end = window(document, args.prompt_tokens, args.prefill_tokens,
                                                  args.start_fraction)
            reference = [int(token) for token in document[1 + text_end:1 + text_end + args.continuation_tokens]]
            if not reference:
                raise ValueError(f"Document {dataset.metadata[index].get('id')} has no tokens after the prompt window")
            document_id = str(dataset.metadata[index].get("id"))
            started = time.monotonic()
            greedy = decode_from_prompt(model, ids, device, greedy_decoding, args.max_new_tokens,
                                        memory=True, document_id=document_id)
            sampled = decode_from_prompt(model, ids, device, {**base_decoding, "seed": args.seed + index + 1},
                                         args.max_new_tokens, memory=True, document_id=document_id) \
                if args.decoding != "greedy" and args.temperature > 0 else greedy
            prefill_seconds = round(time.monotonic() - started, 3)
            scores = {"memory_on": score_continuation(model, ids + reference, prompt_length, device, memory=True,
                                                      document_id=document_id),
                      "memory_off": None if args.no_ablation else
                      score_continuation(model, ids + reference, prompt_length, device, memory=False,
                                         document_id=document_id)}
            total_loss += scores["memory_on"]["loss"] * scores["memory_on"]["tokens"]
            total_tokens += scores["memory_on"]["tokens"]
            if scores["memory_off"]:
                total_off_loss += scores["memory_off"]["loss"] * scores["memory_off"]["tokens"]
            metadata = dataset.metadata[index]
            nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
            report["samples"].append({
                "document_id": document_id, "title": metadata.get("title") or nested.get("title"),
                "metadata": nested or {key: value for key, value in metadata.items() if key != "text"},
                "context_tokens": prompt_length, "reference_tokens": len(reference),
                "prompt_text": clip_middle(tokenizer.decode(ids), args.markdown_characters),
                "reference_text": clip_middle(tokenizer.decode(reference), args.markdown_characters),
                "continuations": {
                    "greedy": {**greedy, "text": tokenizer.decode(greedy["tokens"])},
                    "sampled": {**sampled, "text": tokenizer.decode(sampled["tokens"]),
                                "same_as_greedy": sampled is greedy},
                },
                "scores": scores,
                "memory": {"capacity": model.config.memory_capacity, "chunk_size": model.config.chunk_size,
                           "memory_entries": greedy["memory_entries"], "pending_tokens": greedy["pending_tokens"],
                           "position": greedy["position"], "document_tokens": int(len(document))},
                "seconds": {"prefill": prefill_seconds, "greedy": greedy["seconds"], "sampled": sampled["seconds"]},
            })
            print(json.dumps({"event": "sample", "document_id": document_id, "context_tokens": prompt_length,
                              "reference_tokens": len(reference),
                              "ce_memory_on": round(scores["memory_on"]["loss"], 4),
                              "ce_memory_off": None if not scores["memory_off"] else round(scores["memory_off"]["loss"], 4),
                              "seconds": prefill_seconds}), flush=True)
    mean = total_loss / total_tokens if total_tokens else None
    report["aggregate"] = {"documents": len(report["samples"]), "continuation_tokens": total_tokens,
                           "continuation_loss": mean,
                           "continuation_perplexity": float(torch.exp(torch.tensor(mean))) if mean else None,
                           "memory_off_loss": (total_off_loss / total_tokens
                                               if total_tokens and not args.no_ablation else None)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "samples.md").write_text(render_markdown(report), encoding="utf-8")
    for number, item in enumerate(report["samples"], start=1):
        stem = f"{number:02d}_" + "".join(c if c.isalnum() or c in "-_." else "-" for c in item["document_id"])
        for kind, text in (("greedy", item["continuations"]["greedy"]["text"]),
                           ("sampled", item["continuations"]["sampled"]["text"]),
                           ("reference", item["reference_text"])):
            (out / f"{stem}.{kind}.txt").write_text(item["prompt_text"] + text, encoding="utf-8")
    print(json.dumps({"event": "wrote", "markdown": str(out / "samples.md"), "json": str(out / "samples.json"),
                      "continuation_loss": report["aggregate"]["continuation_loss"],
                      "memory_off_loss": report["aggregate"]["memory_off_loss"]}))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Prepared corpus directory (contains manifest.json)")
    p.add_argument("--split", choices=["val", "test", "train"], default="val",
                   help="Prefer val or test; train is only for debugging")
    p.add_argument("--out", required=True, help="Directory for samples.md, samples.json and text files")
    p.add_argument("--documents", type=int, default=3)
    p.add_argument("--select", choices=["first", "random", "longest", "shortest"], default="first")
    p.add_argument("--document-offset", type=int, default=0)
    p.add_argument("--document-ids", help="Comma-separated document ids; overrides --documents and --select")
    p.add_argument("--prompt-tokens", type=int, default=192)
    p.add_argument("--prefill-tokens", type=int, default=0,
                   help="Real tokens of the same document streamed before the prompt (long-memory probe)")
    p.add_argument("--start-fraction", type=float, default=0.0,
                   help="Where in the document the streamed window starts, in [0, 1)")
    p.add_argument("--continuation-tokens", type=int, default=256,
                   help="Real held-out tokens to score and print as reference")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--decoding", choices=["both", "greedy", "sampled"], default="both")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-ablation", action="store_true", help="Skip the retrieval-off scoring pass (faster)")
    p.add_argument("--markdown-characters", type=int, default=2000, help="Per-block preview limit in samples.md")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    return p


def main():
    generate_samples(parser().parse_args())


if __name__ == "__main__":
    main()
