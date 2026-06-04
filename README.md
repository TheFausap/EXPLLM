# QALF: Quantum Associative Language Field

QALF is a small proof-of-concept language model that uses complex Hilbert-space
states, density-matrix context, entangled relation operators, and Born-style
decoding. It is intentionally not a transformer, RNN, SSM, or wrapper around
pretrained model weights.

The current target is conceptual evidence: a model that can train locally,
generate short coherent replies, and provide enough diagnostics to support a
paper about the idea.

## Environment

The user-provided Conda environment is named `EXPLLM`.

```bash
conda run -n EXPLLM python -m qalf.train --data data/seed_corpus.jsonl --out runs/qalf_poc
conda run -n EXPLLM python -m qalf.eval --checkpoint runs/qalf_poc/model.pt
conda run -n EXPLLM python -m qalf.chat --checkpoint runs/qalf_poc/model.pt
conda run -n EXPLLM python -m unittest discover -s tests
```

QALF uses CUDA automatically when `torch.cuda.is_available()` succeeds and falls
back to CPU otherwise.

## Larger Dataset

TinyStories is a useful next corpus because it was designed for very small
language models and simple coherent English.

```bash
conda run -n EXPLLM python -m qalf.prepare_text --source tinystories-valid --out data/tinystories_qalf.jsonl --max-examples 2000
conda run -n EXPLLM python -m qalf.train --data data/tinystories_qalf.jsonl --out runs/qalf_tinystories_small --device auto --dimension 48 --context-size 32 --vocab-size 2500 --epochs 15 --batch-size 256 --lr 0.012 --max-windows 30000 --attractor-limit 300 --log-every 3
conda run -n EXPLLM python -m qalf.eval --checkpoint runs/qalf_tinystories_small/model.pt --data data/tinystories_qalf.jsonl --out runs/qalf_tinystories_small/eval_compact.json --attractor-data data/tinystories_qalf.jsonl
conda run -n EXPLLM python -m qalf.chat --checkpoint runs/qalf_tinystories_small/model.pt --attractor-data data/tinystories_qalf.jsonl
```

Verified CUDA run on the GTX 1660:

```bash
conda run -n EXPLLM python -m qalf.train --data data/tinystories_qalf.jsonl --out runs/qalf_tinystories --device cuda --dimension 96 --context-size 48 --vocab-size 8000 --epochs 40 --batch-size 256 --lr 0.01 --max-windows 200000
conda run -n EXPLLM python -m qalf.eval --checkpoint runs/qalf_tinystories/model.pt --data data/tinystories_qalf.jsonl --out runs/qalf_tinystories/eval.json --attractor-data data/tinystories_qalf.jsonl --eval-batch-size 256
```

If the normal sandbox runner cannot mount GPU devices, run these commands from the CUDA-visible shell or with approved unsandboxed execution.

## DGX Spark Higher-Order Run

This run enables sparse trigram associative memory and writes train/eval job logs
as JSONL so results can be shared back into Codex.

```bash
conda run -n EXPLLM python -m qalf.train \
  --data data/tinystories_qalf.jsonl \
  --out runs/qalf_dgx_trigram \
  --device cuda \
  --dimension 192 \
  --context-size 96 \
  --vocab-size 16000 \
  --epochs 60 \
  --batch-size 512 \
  --lr 0.006 \
  --max-windows 1000000 \
  --relations 8 \
  --trigram-top-k 64 \
  --trigram-min-count 2 \
  --trigram-strength 0.9 \
  --attractor-limit 1000 \
  --log-every 5 \
  --log-file runs/qalf_dgx_trigram/train.jsonl

conda run -n EXPLLM python -m qalf.eval \
  --checkpoint runs/qalf_dgx_trigram/model.pt \
  --data data/tinystories_qalf.jsonl \
  --out runs/qalf_dgx_trigram/eval.json \
  --attractor-data data/tinystories_qalf.jsonl \
  --eval-batch-size 1024 \
  --log-file runs/qalf_dgx_trigram/eval.jsonl

conda run -n EXPLLM python -m qalf.eval \
  --checkpoint runs/qalf_dgx_trigram/model.pt \
  --data data/tinystories_qalf.jsonl \
  --out runs/qalf_dgx_trigram/eval_raw.json \
  --no-attractor \
  --eval-batch-size 1024 \
  --log-file runs/qalf_dgx_trigram/eval_raw.jsonl
```

Share `runs/qalf_dgx_trigram/train.jsonl`, `eval.json`, and `eval_raw.json`
after the run. If memory is tight, reduce `--batch-size` first, then
`--max-windows`. If training is too fast and underuses the DGX, increase
`--dimension` to `256` and `--max-windows` to all available windows.
