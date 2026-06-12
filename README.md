# QALF: Quantum Associative Language Field

QALF is a small proof-of-concept language model that uses complex Hilbert-space
states, density-matrix context, entangled relation operators, and Born-style
decoding. It is intentionally not a transformer, RNN, SSM, or wrapper around
pretrained model weights.

The current target is conceptual evidence: a model that can train locally,
generate short coherent replies, and provide enough diagnostics to support a
paper about the idea.

## Environment

The Conda environment is named `EXPLLM`.

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

## DGX Spark Heavy Data Run

The first DGX run used little RAM because `data/tinystories_qalf.jsonl` contains
only 2,000 examples and produced only about 202k windows. To use the DGX Spark
128 GB RAM, prepare a much larger TinyStories-train subset first. The updated
trainer logs estimated memory for window tensors, bigram memory, and trigram
memory.

Start here:

```bash
conda run -n EXPLLM python -m qalf.prepare_text \
  --source tinystories-train \
  --out data/tinystories_train_100k_qalf.jsonl \
  --max-examples 100000 \
  --prompt-tokens 24 \
  --reply-tokens 128 \
  --log-file runs/qalf_dgx_heavy/prep.jsonl

conda run -n EXPLLM python -m qalf.train \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_dgx_heavy \
  --device cuda \
  --dimension 384 \
  --context-size 128 \
  --vocab-size 32000 \
  --epochs 30 \
  --batch-size 1024 \
  --lr 0.008 \
  --lr-schedule warmup-cosine \
  --warmup-fraction 0.03 \
  --min-lr-ratio 0.15 \
  --max-windows 5000000 \
  --relations 12 \
  --trigram-top-k 96 \
  --trigram-min-count 2 \
  --trigram-strength 1.0 \
  --bigram-strength 0.35 \
  --entropy-weight 0.02 \
  --component-diversity-weight 0.1 \
  --component-diversity-target 0.05 \
  --attractor-limit 2000 \
  --log-every 2 \
  --log-file runs/qalf_dgx_heavy/train.jsonl

conda run -n EXPLLM python -m qalf.eval \
  --checkpoint runs/qalf_dgx_heavy/model.pt \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_dgx_heavy/eval_raw.json \
  --no-attractor \
  --eval-batch-size 2048 \
  --log-file runs/qalf_dgx_heavy/eval_raw.jsonl
```

Expected memory pressure is mostly from `contexts`: roughly
`max_windows * context_size * 8` bytes before smaller side tensors. With
5,000,000 windows and context 128, the context tensor alone is about 4.8 GiB;
Python overhead is now reduced by preallocating tensors directly. The dense
bigram prior at vocab 32k is about 4 GiB. This should still leave plenty of room
on a 128 GB DGX Spark. If memory remains low and training is stable, raise
`--max-windows` to `10000000`, then raise `--dimension` to `512`.

## Resuming Training

Training now supports periodic resumable checkpoints. Add `--save-every N` to a
long run; this writes `checkpoint_epoch_N.pt` files containing model weights,
optimizer state, and the completed epoch. Final `model.pt` also includes training
state.

Example long run options:

```bash
--save-every 2 \
--log-file runs/qalf_dgx_heavy/train.jsonl
```

Resume from the latest checkpoint and set `--epochs` to the final target epoch,
not the number of additional epochs:

```bash
conda run -n EXPLLM python -m qalf.train \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_dgx_heavy \
  --resume runs/qalf_dgx_heavy/checkpoint_epoch_10.pt \
  --device cuda \
  --epochs 30 \
  --batch-size 1024 \
  --lr 0.004 \
  --save-every 2 \
  --log-file runs/qalf_dgx_heavy/train_resume.jsonl
```

If a run was started without `--save-every`, it can resume only after final
`model.pt` has been written. An interrupted run with no checkpoint cannot be
resumed.

## Learning Rate Scheduling

Training defaults to a static learning rate for compatibility, but long DGX runs
should use warmup plus cosine decay. The schedule is step-based, logs `lr` at
each epoch record, and stores `global_step` in checkpoints for resume.

Recommended starting point for the heavy run:

```bash
--lr 0.008 \
--lr-schedule warmup-cosine \
--warmup-fraction 0.03 \
--min-lr-ratio 0.15
```

If the first two epochs are unstable or loss spikes, lower `--lr` to `0.006`.
If the loss still plateaus early and VRAM is healthy, try `--batch-size 2048`
with `--lr 0.01`. For exact resume behavior, keep dataset, `--max-windows`,
`--batch-size`, and `--epochs` consistent with the original scheduled run.

## QALF-Mixed DGX Run

QALF-Mixed replaces the previous rank-one context with a mixture of several
phase-weighted context components. In logs, early `purity_mean` should be below
`1.0`, but the stronger sanity check is that `component_overlap_mean` stays low
and `density_effective_rank` does not collapse back to `1.0`. If purity rises
toward `1.0` while `component_entropy` remains high, the components have aligned
and the run needs component-diversity regularisation.

Short comparison run first:

```bash
conda run -n EXPLLM python -m qalf.train \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_mixed_compare \
  --device cuda \
  --dimension 512 \
  --context-size 128 \
  --components 6 \
  --vocab-size 32000 \
  --epochs 8 \
  --batch-size 2048 \
  --lr 0.004 \
  --lr-schedule constant \
  --max-windows 10000000 \
  --relations 12 \
  --trigram-top-k 96 \
  --trigram-min-count 2 \
  --trigram-strength 1.0 \
  --bigram-strength 0.35 \
  --entropy-weight 0.02 \
  --component-diversity-weight 0.1 \
  --component-diversity-target 0.05 \
  --attractor-limit 2000 \
  --save-every 2 \
  --log-every 1 \
  --log-file runs/qalf_mixed_compare/train.jsonl
```

If the comparison run beats the rank-one curve or gives better raw samples, run
a longer version:

```bash
conda run -n EXPLLM python -m qalf.train \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_mixed_dgx \
  --device cuda \
  --dimension 768 \
  --context-size 160 \
  --components 8 \
  --vocab-size 32000 \
  --epochs 20 \
  --batch-size 2048 \
  --lr 0.004 \
  --lr-schedule constant \
  --max-windows 10000000 \
  --relations 16 \
  --trigram-top-k 128 \
  --trigram-min-count 2 \
  --trigram-strength 1.0 \
  --attractor-limit 2000 \
  --save-every 2 \
  --log-every 1 \
  --log-file runs/qalf_mixed_dgx/train.jsonl
```

Evaluate raw generation first; the main goal is not lower attractor-backed loss,
but better `--no-attractor` samples:

```bash
conda run -n EXPLLM python -m qalf.eval \
  --checkpoint runs/qalf_mixed_compare/model.pt \
  --data data/tinystories_train_100k_qalf.jsonl \
  --out runs/qalf_mixed_compare/eval_raw.json \
  --no-attractor \
  --eval-batch-size 1024 \
  --log-file runs/qalf_mixed_compare/eval_raw.jsonl
```
