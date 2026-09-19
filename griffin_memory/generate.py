"""Autoregressive generation with persistent RG-LRU, KV and chunk memories."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from .config import ModelConfig
from .model import GriffinMemoryLM
from .tokenizer import Tokenizer, BOS, EOS, PAD
from .train import amp_context, choose_device, load_checkpoint


@torch.no_grad()
def prefill(model, tokens, state=None):
    """Respect memory boundaries even for a prompt longer than a single chunk."""
    logits = None
    offset = 0
    while offset < tokens.shape[1]:
        position = state.position if state is not None else 0
        count = min(model.config.chunk_size - position % model.config.chunk_size, tokens.shape[1] - offset)
        logits, state = model(tokens[:, offset:offset + count], state,
                              document_ids=[f"generation:{i}" for i in range(tokens.shape[0])])
        offset += count
    if logits is None:
        raise ValueError("Prompt must contain at least BOS")
    return logits[:, -1], state


def sample(logits, temperature, top_k, top_p):
    logits = logits.float().clone()
    logits[:, [PAD, BOS]] = -float("inf")
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits /= temperature
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits.masked_fill_(logits < threshold, -float("inf"))
    if top_p < 1:
        ordered, ids = logits.sort(descending=True, dim=-1)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative - ordered.softmax(-1) >= top_p
        ordered.masked_fill_(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, ids, ordered)
    return torch.multinomial(logits.softmax(-1), num_samples=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", help="Write full prompt and continuation as UTF-8")
    p.add_argument("--memory-out", help="Export bounded memory chunks, vectors, and metadata as JSON")
    args = p.parse_args()
    if args.max_new_tokens < 0 or args.temperature < 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        p.error("Invalid generation settings")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        p.error("fp16 requires CUDA")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("bf16 is not supported on this GPU")
    checkpoint = load_checkpoint(args.checkpoint)
    config = ModelConfig(**checkpoint["config"])
    model = GriffinMemoryLM(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    # Tokenizer embedded in checkpoint makes copied checkpoints self-contained.
    with tempfile.TemporaryDirectory() as temp:
        tokenizer_path = Path(temp) / "tokenizer.json"
        tokenizer_path.write_text(checkpoint["tokenizer_json"])
        tokenizer = Tokenizer.load(tokenizer_path)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    ids = [BOS] + tokenizer.encode(prompt)
    torch.manual_seed(args.seed)
    with torch.inference_mode(), amp_context(device, args.precision):
        logits, state = prefill(model, torch.tensor([ids], device=device))
        for _ in range(args.max_new_tokens):
            next_token = sample(logits, args.temperature, args.top_k, args.top_p)
            token_id = int(next_token.item())
            if token_id == EOS:
                break
            ids.append(token_id)
            # Consume the sampled token so all three caches also include it.
            logits, state = model(next_token, state, document_ids=["generation:0"])
            logits = logits[:, -1]
    text = tokenizer.decode(ids)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    if args.memory_out:
        entries = [{"text": tokenizer.decode(e.token_ids), "token_ids": e.token_ids,
                    "metadata": e.metadata, "key": e.key.tolist(), "value": e.value.tolist()}
                   for e in state.banks[0].entries]
        Path(args.memory_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.memory_out).write_text(json.dumps({"position": state.position,
              "pending_tokens": state.pending_tokens[0], "entries": entries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
