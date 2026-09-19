"""Evaluate a checkpoint with fresh document-local memories and token-weighted CE."""
import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .data import DocumentDataset, dataset_manifest
from .model import GriffinMemoryLM
from .train import choose_device, evaluate, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0, help="0 evaluates the full split")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()
    if args.batch_size <= 0 or args.max_batches < 0:
        p.error("batch-size must be positive and max-batches nonnegative")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        p.error("fp16 requires CUDA")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("bf16 is not supported on this GPU")
    checkpoint = load_checkpoint(args.checkpoint)
    dataset_manifest(args.data)
    if json.loads((Path(args.data) / "tokenizer.json").read_text()) != json.loads(checkpoint["tokenizer_json"]):
        p.error("Dataset tokenizer differs from the checkpoint tokenizer")
    if not (Path(args.data) / f"{args.split}.offsets.npy").is_file():
        p.error(f"The dataset does not include split {args.split!r}")
    dataset = DocumentDataset(args.data, args.split)
    if not len(dataset):
        p.error("The selected split has no documents")
    model = GriffinMemoryLM(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    options = {"batch_size": args.batch_size, "seed": checkpoint["options"]["seed"], "precision": args.precision}
    metrics = evaluate(model, dataset, options, device, args.max_batches)
    metrics = {key.replace("val_", f"{args.split}_", 1): value for key, value in metrics.items()}
    record = {"split": args.split, "checkpoint_step": checkpoint["step"], **metrics}
    print(json.dumps(record, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
