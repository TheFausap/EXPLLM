# PG-19 full-run runbook (single device)

This is the practical, end-to-end guide for running the `griffin_memory` model on
**PG-19**: prepare whole books, plan a token budget, probe throughput, train,
generate held-out samples, evaluate and read the resulting report.

It documents `griffin_memory/scripts/pg19_full_run.sh` plus the two modules it
drives: `griffin_memory/run_report.py` (`plan` / `estimate` / `build`) and
`griffin_memory/generate_samples.py` (held-out generation reports). See
[DGX_SPARK.md](DGX_SPARK.md) for the hardware/scaling discussion and
[README.md](README.md) for the architecture itself.

**Scope warning, up front.** A "full run over PG-19" here means one file-backed
document stream over a bounded subset of whole books with the reference
implementation in this repository. It is not a reproduction of the PG-19
benchmark numbers, and it does not by itself establish production-grade
long-form coherence. The dataset authors' metric is a word-normalized
perplexity; this implementation reports **subword-token** cross-entropy.

---

## 1. What you get

One command runs the whole pipeline and leaves this layout:

```text
RUN_DIR/
  driver/          plan.json, estimate.json, commands.log, pilot/ (throughput probe)
  corpus/          manifest.json, tokenizer.json, train|val|test.{bin,offsets.npy,metadata.jsonl}
  train/           last.pt, best.pt, train.jsonl, run.json, config.json, tokenizer.json
  samples/         samples.md, samples.json, NN_<id>.{greedy,sampled,reference}.txt
  eval/            val.json, test.json (test only with --eval-test)
  run_report.md    corpus + training + validation + samples in one document
```

`run_report.md` is the deliverable: corpus provenance, budget versus actual,
throughput, validation curve, full held-out generations side by side with the
real continuation, and the exact commands that produced them.

Put `RUN_DIR` wherever you keep large outputs. `griffin_memory/artifacts/` and
any `artifacts/` directory at the repository root are already ignored by Git;
checkpoints, samples and prepared corpora do not belong in a commit.

## 2. Before the first run

1. **Environment.** Python 3.11+, PyTorch 2.5+. Install the corpus extras
   (`tokenizers`, `huggingface-hub`) only if you use BPE or the preset downloaders:
   `python -m pip install -r griffin_memory/requirements-data.txt`.
2. **Network.** Preparation needs HTTPS access to Hugging Face (TinyStories),
   `storage.googleapis.com` (PG-19 books) and `raw.githubusercontent.com`
   (pinned metadata). Never disable TLS verification to get around a failure.
3. **Disk.** Reserve roughly 20 GB for a 1,000-book starter run (download cache,
   raw text, uint32 token streams, checkpoints, pilot copy). More is better;
   the cache is reusable.
4. **Licence.** PG-19 is distributed under **Apache-2.0** and the authors warn
   about historical style and bias. A pre-1919 publication date is not a
   worldwide copyright clearance for the underlying works. Check the rights in
   your jurisdiction and do not redistribute generated or reference text from
   books whose rights you cannot confirm.
5. **Test-split discipline.** `--include-test` downloads the 100 official test
   books and prepares them as a separate split. Training, model selection and
   periodic validation never touch it. Evaluate it once, after you freeze the
   experiment (`--eval-test`).

## 3. Quick start

```bash
# 0. verification pass on the bundled examples (CPU, seconds, no downloads)
python -m griffin_memory.prepare --input griffin_memory/examples/stories.jsonl \
  --out runs/smoke-corpus --val-fraction 0.25
bash griffin_memory/scripts/pg19_full_run.sh runs/smoke \
  --data runs/smoke-corpus --profile smoke --max-steps 40

# 1. the real thing: 1,000 PG-19 books, 3 epochs of budget, 4 samples
bash griffin_memory/scripts/pg19_full_run.sh runs/pg19-full \
  --dataset pg19 --train-limit 1000 --val-limit 50 --include-test \
  --token-budget 300000000 --samples 4 --prefill-tokens 4096

# 2. continue after an interruption (never re-plans the schedule)
bash griffin_memory/scripts/pg19_full_run.sh runs/pg19-full --only train --resume

# 3. rebuild the report from whatever finished
bash griffin_memory/scripts/pg19_full_run.sh runs/pg19-full --only report
```

`--dry-run` prints every command without touching the filesystem; use it to
check a plan before starting a multi-hour job. `PYTHON` selects the interpreter
(defaults to `python`), which is how you point the driver at a virtualenv or a
container interpreter.

## 4. Stage by stage

| Stage | What it runs | Skip when |
|---|---|---|
| `env` | Torch/CUDA/BF16 check plus a real matrix forward/backward | Never on a new machine |
| `prepare` | `prepare_corpus` (presets) or `prepare` (local JSONL), then a prepared corpus is reused as-is | Reusing `--data` |
| `plan` | `run_report plan` → `driver/plan.json` (targets → steps) | Reusing an existing plan |
| `pilot` | 100-update throughput probe into `driver/pilot`, then `run_report estimate` | `--pilot-steps 0` |
| `train` | `train` into `RUN_DIR/train` with the planned `--max-steps` | Re-running samples only |
| `sample` | `generate_samples` into `RUN_DIR/samples` from `best.pt` (or `last.pt`) | You only want metrics |
| `eval` | `evaluate` on `val` (and `test` with `--eval-test`) | A pilot run |
| `report` | `run_report build` → `RUN_DIR/run_report.md` | — |

Select with `--only prepare,plan,train` or `--skip pilot,sample`. The `env`,
`prepare` and `plan` stages can be skipped independently; every stage fails
loudly rather than silently skipping inputs.

### 4.1 Corpus preparation

```bash
# what the driver runs for you
python -m griffin_memory.prepare_corpus \
  --dataset pg19 --out griffin_memory/artifacts/pg19-1k \
  --train-limit 1000 --val-limit 50 --include-test \
  --tokenizer bpe --vocab-size 16000 --tokenizer-max-characters 20000000 \
  --cache-dir griffin_memory/artifacts/downloads --workers 4 --seed 42
```

* One book stays **one document**: not split per chapter, not shuffled at chunk
  level, not truncated. The 256-token chunks and all three memory tiers stream
  the book from its first token to its last.
* Only the selected books are downloaded. Listings and books are cached and
  pinned to a GCS object generation plus the published MD5; an interrupted
  preparation reuses every verified book.
* Splits are official: no resplitting, and held-out text never trains BPE.
  Exact duplicates are removed with test > validation > train priority.
* The BPE fit is bounded to 20M characters sampled across training documents;
  `--tokenizer-max-characters 0` removes the cap.
* `--tokenizer-file` reuses a vocabulary, which keeps token ids compatible with
  an existing checkpoint. It does **not** make a different corpus resumable.

### 4.2 Plan the budget

`plan` converts the corpus manifest and your batch shape into steps:

```bash
python -m griffin_memory.run_report plan \
  --data griffin_memory/artifacts/pg19-1k --config griffin_memory/configs/small.json \
  --batch-size 1 --accumulation-steps 16 --token-budget 300000000 \
  --out runs/pg19-plan.json
```

`targets / (batch × accumulation × chunk_size)` gives the steps for a full pass;
`--token-budget` (planned prediction targets) or `--max-steps` override the
epoch-derived default. Two honest caveats are printed with the plan:

* the logged token count is **lower** than `steps × tokens_per_update`, because
  padded lanes and short final chunks contribute no targets;
* a corpus below ~200M targets cannot produce fluent modern prose with a 10M
  model, whatever the step count says.

**Budget arithmetic.** Steps are cheap to compute; time is not. Measure it (§4.3)
rather than guessing, because throughput depends on document length, chunk size,
precision and how much of the machine you have.

| Scenario | Train books | Rough targets | Updates at 4,096 targets/update | Purpose |
|---|---:|---:|---:|---|
| Toy verification (`--profile smoke`) | bundled corpus | ~50k | 60 | Pipeline check only |
| Pilot | 50 | ~5M | ~1,200 | Throughput + early-loss check |
| Starter | 250 | ~25M | ~6,100 | First readable, locally coherent output |
| Full starter budget | 1,000 × 3 epochs | ~300M | ~73,000 | The realistic single-device target |

Targets are order-of-magnitude planning figures; a 16k-vocabulary PG-19 book is
tens to hundreds of thousands of tokens and lengths vary by a large factor. The
manifest (`manifest.json → stats`) and `plan.json` carry the measured numbers for
your subset, so read those instead of trusting the table.

For reference, this repository's sandbox (2 x86 cores, FP32, no GPU, PyTorch
2.5.1) measured **~500 targets/s** for `configs/small.json` (chunk 256, batch 1 x
accumulation 4) and **~2,070 targets/s** for `configs/smoke.json` (chunk 32). At
500 targets/s the 300M-target budget is ~167 hours; the 25M starter budget is
~14 hours. A DGX Spark should be far faster, but its throughput is exactly what
the pilot measures — do not extrapolate from these two CPU numbers.

### 4.3 Pilot, then the long run

The pilot is a real 100-update training run in `RUN_DIR/driver/pilot` with the
same batch shape and precision; `run_report estimate` then reads its log:

```bash
python -m griffin_memory.run_report estimate \
  --log RUN_DIR/driver/pilot/train.jsonl --planned-steps 73000 --out RUN_DIR/driver/estimate.json
```

It reports the median targets/s of the last 20 logged records, seconds per
update, and the projected wall-clock hours for the planned budget. Decide from
that number, not from hope.

```bash
python -m griffin_memory.train \
  --data RUN_DIR/corpus --config griffin_memory/configs/small.json --out RUN_DIR/train \
  --device cuda --precision bf16 --batch-size 1 --accumulation-steps 16 \
  --max-steps 73000 --epochs 3 --warmup-steps 500 --learning-rate 0.0003 \
  --weight-decay 0.1 --clip-grad 1.0 \
  --eval-every 500 --eval-batches 64 --save-every 250 --log-every 10
```

The run stops at whichever comes first: `--max-steps` updates or `--epochs`
passes. AdamW uses `(0.9, 0.95)`, linear warmup and cosine decay to
`min_lr_ratio`; the schedule is fixed by `--max-steps`, so **plan the full budget
before starting**. To continue an experiment, resume the same run directory:

```bash
python -m griffin_memory.train --data RUN_DIR/corpus --out RUN_DIR/train \
  --resume RUN_DIR/train/last.pt --device cuda --threads 4
```

Saved hyperparameters are restored and conflicting overrides are rejected;
switching corpora or tokenizers via `--resume` is refused. To rehearse an
interruption without changing the schedule, add `--stop-after-steps 100` — it is
not the same as `--max-steps 100`. A completed budget is a finished experiment:
start a new run (and a new run directory) for a larger one.

### 4.4 Generation samples (the part that actually shows coherence)

```bash
python -m griffin_memory.generate_samples \
  --checkpoint RUN_DIR/train/best.pt --data RUN_DIR/corpus --split val \
  --out RUN_DIR/samples --documents 4 --select longest \
  --prompt-tokens 192 --prefill-tokens 4096 --start-fraction 0.25 \
  --max-new-tokens 256 --continuation-tokens 256 \
  --temperature 0.8 --top-k 40 --top-p 0.95 --device cuda --precision bf16
```

For each document it writes:

* the **real held-out prompt** (exactly the tokenizer's ids, no re-encoding);
* a **greedy** and a **sampled** continuation from the same context;
* the **real continuation** the model was trying to predict;
* **continuation CE**, teacher-forced, on the real next tokens — once with the
  retrieval tier active and once with it disabled (`--no-ablation` skips the
  second pass);
* memory diagnostics: chunks resident in the retrieval bank, capacity, pending
  tokens, streamed positions.

`--prefill-tokens` is the long-memory probe: that many real tokens of the same
book are streamed *before* the prompt, so the retrieval bank already holds
earlier chapters and older chunks may have been evicted by FIFO. With the small
preset, capacity is 256 chunks × their token span, so a book longer than roughly
65k tokens cannot be retrieved whole — earlier material survives only in the
recurrent state. If you want longer explicit memory, raise `memory_capacity` in
the config and accept the CPU cost of a larger ANN index; changing the config
means a fresh run.

Reading the numbers honestly:

* CE/perplexity here are **subword-token** metrics on a handful of documents.
  They are not the dataset's word-normalized perplexity and not a quality score.
* The `mem off` column isolates *this* implementation's retrieval tier on *this*
  prompt. A small gap is normal at small budgets; it is not a retrieval-recall
  benchmark.
* Read the text. A model can lower CE while still drifting; only the side-by-side
  continuation shows whether the prose holds together.

### 4.5 Evaluation

```bash
# full official validation split, no batch cap
python -m griffin_memory.evaluate --checkpoint RUN_DIR/train/best.pt --data RUN_DIR/corpus \
  --split val --device cuda --precision bf16 --batch-size 1 --max-batches 0

# the untouched test books, once, after freezing the experiment
python -m griffin_memory.evaluate --checkpoint RUN_DIR/train/best.pt --data RUN_DIR/corpus \
  --split test --device cuda --precision bf16 --batch-size 1 --max-batches 0
```

`--max-batches 0` means the whole split; a positive cap evaluates only an initial
prefix of the groups, which for PG-19 often means part of a single book. The
driver's periodic `--eval-batches 64` is a progress signal, not a final number.
Report the vocabulary size next to any perplexity, and never compare perplexities
across tokenizers.

### 4.6 The run report

```bash
python -m griffin_memory.run_report build --run-dir RUN_DIR --out RUN_DIR/run_report.md
```

It collects `corpus`/`driver`/`train`/`samples`/`eval` (overridable with `--data`,
`--plan`, `--estimate`, `--samples`, `--eval`, `--run-json`, `--config`, `--log`,
`--commands`) and fails if the run has no `run.json` or training log. Structure:

```text
# Run report — pg19-full
Key outputs: .../last.pt (checkpoints), .../samples.json (generation samples), ...
## Corpus        <- documents, tokens, targets, provenance, filter counts
## Training      <- device, batch shape, LR schedule, observed throughput, cursor
## Validation    <- val loss/perplexity per evaluation
## Commands      <- the exact commands, from driver/commands.log
## Generation samples   <- samples.md in full (prompt / greedy / sampled / reference)
## How to read this
```

## 5. Manual pipeline (without the driver)

```bash
python -m griffin_memory.prepare_corpus --dataset pg19 --out artifacts/pg19-1k \
  --train-limit 1000 --val-limit 50 --include-test --tokenizer bpe --vocab-size 16000 \
  --tokenizer-max-characters 20000000 --workers 4 --seed 42
python -m griffin_memory.run_report plan --data artifacts/pg19-1k \
  --config griffin_memory/configs/small.json --batch-size 1 --accumulation-steps 16 \
  --token-budget 300000000 --out runs/pg19-plan.json
python -m griffin_memory.train --data artifacts/pg19-1k --config griffin_memory/configs/small.json \
  --out runs/pg19 --device cuda --precision bf16 --batch-size 1 --accumulation-steps 16 \
  --max-steps 73000 --epochs 3 --eval-every 500 --save-every 250
python -m griffin_memory.generate_samples --checkpoint runs/pg19/best.pt --data artifacts/pg19-1k \
  --out runs/pg19/samples --documents 4 --prefill-tokens 4096
python -m griffin_memory.evaluate --checkpoint runs/pg19/best.pt --data artifacts/pg19-1k \
  --split val --max-batches 0
python -m griffin_memory.run_report build --run-dir runs/pg19 --data artifacts/pg19-1k
```

With local, already-downloaded JSONL, replace the first command with
`prepare --input train.jsonl --validation-input validation.jsonl [--test-input test.jsonl]`,
or use the driver's `--train-jsonl/--val-jsonl/--test-jsonl`. Note that `prepare`
does not deduplicate; `prepare_corpus` does.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA was requested but unavailable` | The driver defaults to `--device cuda`. On CPU use `--device cpu --precision fp32`; on a GPU host fix the container/driver rather than installing a generic wheel |
| SSL/TLS failure during preparation | No HTTPS path to HF/GCS/raw.githubusercontent.com. Fix networking; never disable verification. Re-run with the same `--cache-dir` to reuse verified downloads |
| `--data expects an existing prepared corpus` | You pointed `--data` at a directory without `manifest.json`. Drop `--data` to prepare inside the run directory |
| `Cannot resume on a different dataset/tokenizer` | `--resume` is for the same corpus. Cross-corpus weight initialization is not implemented here |
| `Config does not match checkpoint` | The saved architecture is authoritative; do not pass a different `--config` on resume |
| Slow: seconds per update far above the estimate | The recurrent scan and the CPU LSH index are the bottlenecks. Lower `chunk_size`/`window_size`, use `--precision bf16`, or reduce the corpus; do not expect more cores to fix a Python scan |
| `Non-finite training loss` | The previous checkpoint is intact. Lower the learning rate or use the FP16 scaler (`--precision fp16`) |
| Out of memory | Reduce `--batch-size` first, then `--accumulation-steps`, chunk/window size, or `memory_capacity` |
| Empty or absurd samples | Expected below ~10M targets. Check `driver/plan.json`, then use the starter budget; the smoke profile is a mechanics check, not a quality claim |

## 7. Verification status

Verified in this repository's sandbox (2 x86 cores, CPU, PyTorch 2.5.1) while
adding these tools:

* 58 `griffin_memory` tests pass, including 20 new tests for window selection,
  chunk-boundary-safe streaming, chunked-versus-token-by-token CE agreement,
  sample reports, run-report planning/estimation building, and the driver script
  itself (argument errors, dry-run, full offline smoke pipeline).
* A complete driver run over a book-shaped local fixture: prepare → plan → train
  (798 updates) → samples → evaluation → `run_report.md`, all on CPU.
* Pilot + resume paths: `--pilot-steps` producing a throughput estimate, and
  `--only train --resume` continuing an interrupted run.

**Not verified here:** any GPU path (CUDA/BF16/FP16), a live PG-19 download (the
sandbox cannot reach the download hosts), and therefore any claim about real
PG-19 throughput or text quality. Treat the throughput figures above as a CPU
reference point and measure your own with the pilot stage.
