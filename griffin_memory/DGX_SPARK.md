# Dataset starters and single-DGX-Spark workflow

## Recommended order

Start with **TinyStories**, then run a separate **PG-19** experiment. The first
checks language learning cheaply; the second actually exercises document-long
recurrent and retrieval memory. Neither alone is a finished modern creative-writing
corpus.

| Dataset | Why use it here? | Suggested first subset | Main limitation |
|---|---|---|---|
| TinyStories (original release) | Synthetic GPT-3.5/GPT-4 stories with deliberately simple vocabulary; a useful small-model learning check | 100,000 train stories + 2,000 official validation stories | Short, simple stories do not establish long-range coherence |
| PG-19 | Whole Project Gutenberg books published before 1919; designed for long-range language-model evaluation | 1,000 train books + all 50 official validation books; optional 100 test books | Historical style, biases, and mixed genres/nonfiction; not a modern fiction-only corpus |

TinyStories' original train text is about **1.92 GB**, with about **19.4 MB** of
validation text. The script intentionally uses the original files, not a mixture
of the original and V2 releases (their content overlaps). [2](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)
The source card declares **CDLA-Sharing-1.0**; review the source's license and
redistribution obligations before use:
<https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/README.md>.

PG-19 has **28,602 / 50 / 100** train/validation/test books. Its source distribution
is labeled **Apache-2.0**. The authors explicitly caution about historical style
and biases. [5](https://huggingface.co/datasets/deepmind/pg19/blob/main/README.md)
Review rights in the underlying works in your jurisdiction too: a dataset license
or a pre-1919 publication date is not a blanket worldwide copyright clearance.
The authoritative book source used here is DeepMind's public
`gs://deepmind-gutenberg` bucket, linked from
<https://github.com/google-deepmind/pg19>.

These are **starter subset sizes**, not fixed token counts or sufficient training
budgets. Inspect `manifest.json` after preparation for actual documents, tokens,
and prediction targets. Size and learning quality depend on the tokenizer and
selected documents. Start with 100 optimizer updates to measure throughput,
then decide the token/time budget rather than assuming a large run will be fast.

## DGX Spark environment

DGX Spark combines a GB10 Blackwell GPU, a 20-core Arm CPU, and **128 GB shared
unified memory**. That is one shared pool, not 128 GB GPU memory plus another
128 GB CPU memory. NVIDIA recommends its framework containers for current
hardware support. [3](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf)

Use an **ARM64/GB10-compatible NVIDIA PyTorch environment** matched to your DGX OS
and driver. Choose a current Spark-supported NGC PyTorch container tag rather
than copying an old generic x86/CUDA image. Keep the PyTorch build supplied with
that environment; do not replace it with a generic wheel just to prepare data.
All commands below run inside that environment, from the repository root.

```bash
# Additional corpus/tokenizer dependencies only; intentionally does not install torch.
python -m pip install -r griffin_memory/requirements-data.txt

python - <<'PY'
import platform, torch
print(platform.machine(), torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
assert torch.cuda.is_bf16_supported(), 'Check the Spark software stack'
PY
```

Prefer BF16 for the first runs. The model maintains recurrent accumulation and
normalization statistics in FP32. No FlashAttention package, bitsandbytes,
`datasets`, FAISS, or remote dataset-loading code is required. The trainer is
single-device; do not launch it with multi-rank `torchrun`.

**Important performance limit:** the implementation still has a Python recurrent
scan, CPU LSH retrieval, and synchronization overhead. Spark's memory capacity
does not remove those bottlenecks. Start with `configs/small.json` (about **10.7M
parameters at vocabulary 16,000**), profile it, then consider the existing
`medium.json` (~128.2M). These are practical experiment sizes, not measured
Spark throughput/VRAM guarantees. No Spark hardware is available in this sandbox.

## Option 1 — TinyStories (recommended first)

```bash
python -m griffin_memory.prepare_corpus \
  --dataset tinystories \
  --out griffin_memory/artifacts/tinystories-100k \
  --train-limit 100000 --val-limit 2000 \
  --tokenizer bpe --vocab-size 16000 \
  --tokenizer-max-characters 20000000 --seed 42
```

Defaults already match this command. The downloader resolves the HF revision to
an immutable commit, downloads the original train and validation text files via
`huggingface_hub`, records their SHA-256 checksums and source card, then samples
story indices uniformly with a seeded RNG. **Even a small subset downloads the
full two source files once** (roughly 2 GB total). Subsequent runs reuse the cache.
A bounded subset uses two sequential disk scans rather than storing all stories
in RAM. Newline normalization and surrounding story whitespace are removed;
`<|endoftext|>` separators are replaced by the model's BOS/EOS document framing.
TinyStories has no official test split in these files.

Run an initial 100-update pilot:

```bash
bash griffin_memory/scripts/train_spark.sh tinystories \
  griffin_memory/artifacts/tinystories-100k \
  griffin_memory/artifacts/spark-tiny-pilot \
  --max-steps 100 --warmup-steps 10 --eval-batches 16
```

For a fresh longer experiment, omit those overrides and use a new run directory.
The launcher defaults to the small model, BF16, batch size 4, accumulation 4,
chunk length 256, 5,000 updates, and at most 3 epochs. There are at most **4,096
valid target tokens per update**; padding and shorter final chunks reduce that.
It first runs a CUDA/BF16 matrix-multiply forward/backward check.

To use the full original corpus later:

```bash
python -m griffin_memory.prepare_corpus \
  --dataset tinystories --train-limit 0 --val-limit 0 \
  --out griffin_memory/artifacts/tinystories-full \
  --tokenizer-file griffin_memory/artifacts/tinystories-100k/tokenizer.json
```

Reusing the tokenizer keeps token IDs compatible. It does **not** make a changed
corpus compatible with the trainer's strict full-state `--resume`; see below.

## Option 2 — PG-19 (long-memory experiment)

For the complete staged workflow (prepare → plan → pilot → train → samples →
evaluation → `run_report.md`), budget arithmetic and troubleshooting see
**[PG19_RUNBOOK.md](PG19_RUNBOOK.md)**, which wraps the commands below in
`bash griffin_memory/scripts/pg19_full_run.sh RUN_DIR --dataset pg19 ...`.

```bash
python -m griffin_memory.prepare_corpus \
  --dataset pg19 \
  --out griffin_memory/artifacts/pg19-1k \
  --train-limit 1000 --val-limit 50 \
  --tokenizer bpe --vocab-size 16000 \
  --tokenizer-max-characters 20000000 \
  --workers 4 --seed 42 --include-test
```

The script lists the official bucket's original splits, samples book IDs with a
seeded RNG, and downloads **only the selected books**, not the entire corpus.
The metadata CSV is pinned to a Git commit. Each book download is pinned to its
GCS object generation and checked against the published MD5 and byte count.
Verified objects and listings are cached, so rerunning after an interruption
reuses completed downloads. An incomplete book is retried from the beginning.
`--refresh-source` refreshes listings; chosen generations are saved in provenance.

One book remains **one document**. It is not broken into separate training
examples per chapter, shuffled at chunk level, or truncated to a context window.
UTF-8 decoding normalizes line endings, but chapter content is retained.
The existing trainer streams it through 256-token chunks, carrying all memory
tiers until the book ends. With the small preset's 256-entry memory bank, at most
256 completed chunks remain explicitly retrievable; older material survives
only in compressed recurrent state. Long documents are not unlimited exact memory.

```bash
bash griffin_memory/scripts/train_spark.sh pg19 \
  griffin_memory/artifacts/pg19-1k \
  griffin_memory/artifacts/spark-pg19-pilot \
  --max-steps 100 --warmup-steps 10 --eval-batches 16
```

This profile starts at **batch size 1, accumulation 16**: books vary greatly in
length, and the current group-based loader wastes computation padding shorter
books beside a much longer book. Batch 1 avoids that waste; accumulation preserves
up to 4,096 targets per update but does not increase instantaneous GPU parallelism.
After profiling, test batches 2–4 or group documents by length in a future loader.
Do not assume batch 1 fully utilizes Spark.

Without pilot overrides the launcher plans 5,000 updates / at most 3 epochs,
not necessarily a full pass over 1,000 books. The corpus manifest and training
logs show the actual target count and cursor. To try the larger model on a fresh
run, append `--config griffin_memory/configs/medium.json` and re-profile first.

`--include-test` prepares the original test books separately. The training loop
never uses them for loss updates, model selection, or its periodic validation.
Without that flag, test books are not downloaded or prepared. A test split may
be evaluated explicitly **after** choosing the experiment:

```bash
python -m griffin_memory.evaluate \
  --checkpoint griffin_memory/artifacts/spark-pg19-pilot/best.pt \
  --data griffin_memory/artifacts/pg19-1k --split test \
  --device cuda --precision bf16 --batch-size 1 --max-batches 0
```

A small `--eval-batches` pilot checks only a prefix, often part of the first long
book. It is not representative book-wide validation. For a serious comparison,
evaluate all validation documents with `--split val --max-batches 0` before
selecting a model. Report this implementation's **subword-token perplexity** as
such: it is not directly the word-normalized PG-19 benchmark metric described
by the dataset authors. [5](https://huggingface.co/datasets/deepmind/pg19/blob/main/README.md)

## Shared pipeline guarantees and limits

- **No resplitting:** samples remain in their original source split. Held-out
  text never trains BPE or the model. Training hashes all manifest files for
  integrity, but does not load test examples into its training/evaluation stream.
- **Exact deduplication:** a disk-backed SQLite hash index reserves test, then
  validation, then training texts. Same-split duplicates and training copies of
  held-out text are dropped, not reassigned. This can produce fewer documents
  than the requested limit. `sources.json` records counts and policies. It is
  not near-duplicate detection, and filtered subsets are not unchanged benchmark
  distributions. There is no automatic top-up after filtering.
- **Bounded BPE fit:** the default 20-million-character budget is proportionally
  allocated across selected training documents, sampling spans up to 4,096
  characters. It does **not** truncate the encoded training/validation/test
  documents. `--tokenizer-max-characters 0` disables the cap; memory use may rise.
  `--tokenizer byte` avoids tokenizer fitting; `--tokenizer-file` reuses a vocabulary.
- **Document framing:** each story/book receives BOS/EOS; no text from unrelated
  stories/books is concatenated to fake a long context.
- **Reproducibility:** provenance includes source revision/generations, source
  checksums, sampling seed, limits, retained IDs in per-document metadata,
  source license declaration, and filtering counts. Prepared files and provenance
  are SHA-256 checksummed in `manifest.json`.
- **Storage:** default persistent cache is `griffin_memory/artifacts/downloads`.
  Use `--cache-dir /path/on/your/ssd` to move it. Reserve roughly 10–20 GB free
  for starter subsets and scratch space; full PG-19 warrants much more (e.g.
  100 GB spare including scratch/checkpoints). These are planning allowances,
  not measured subset sizes. Raw JSONL and tokenizer spools temporarily coexist.
  Corpus text is processed a document at a time; exceptionally large single
  books still need to fit in RAM for tokenization.
- **Safe outputs:** final datasets appear only after successful preparation.
  Existing output directories are not overwritten. Temporary spools are cleaned
  on ordinary errors/interrupts; completed source cache files remain reusable.
  SIGKILL/power loss can leave hidden temporary directories to remove manually.

For locally downloaded official JSONL files, the generic preparer now also accepts
explicit splits without another network download:

```bash
python -m griffin_memory.prepare \
  --input /path/train.jsonl --validation-input /path/validation.jsonl \
  --test-input /path/test.jsonl \
  --out griffin_memory/artifacts/local-corpus \
  --tokenizer bpe --vocab-size 16000 --tokenizer-max-characters 20000000
```

This low-level command preserves supplied splits but **does not itself deduplicate**
them; the high-level `prepare_corpus` command performs deduplication.

## Checkpoint continuation

The launch script is for **fresh experiments**, with explicit profile defaults.
For continuing the same run, use the underlying trainer directly so its saved
hyperparameters are restored without conflicting profile defaults:

```bash
python -m griffin_memory.train \
  --data griffin_memory/artifacts/tinystories-100k \
  --out griffin_memory/artifacts/spark-tiny-run \
  --resume griffin_memory/artifacts/spark-tiny-run/last.pt \
  --device cuda --threads 4
```

To stop and later continue a planned run, use `--stop-after-steps 100`, not
`--max-steps 100`: the former retains the original LR schedule/budget. A completed
100-update pilot is a finished experiment; start a new run for a larger budget.
TinyStories → PG-19 here means **two independent scratch experiments**. Switching
corpora using `--resume` is intentionally rejected. Weight-only initialization
for a cross-corpus curriculum is not implemented by these scripts.

## Validation status

The corpus adapters have 12 offline tests covering framing, seeded selection,
official split preservation, whole-book metadata, checksum/cache/retry behavior,
GCS listing pagination, deduplication priority, train-only bounded BPE fitting,
explicit test preparation, and a fixture-based end-to-end run. All **58**
`griffin_memory` tests (including 20 for the PG-19 pipeline, generation reports
and the driver script) and the repository's other tests pass on CPU. The driver
itself was exercised end-to-end on a book-shaped local fixture with `--dry-run`,
`--only`, `--pilot-steps` and `--resume`.

Source URLs, filenames, licensing cards, and the official GCS schema were checked
via web retrieval. This sandbox cannot connect to those download hosts through
its Python TLS/network path, so **a live full-corpus download and a run on DGX
Spark have not been verified here**. Do not disable TLS verification to work
around network failures. On the Spark, ensure normal HTTPS access to Hugging
Face/CDN, Google Cloud Storage, and raw.githubusercontent.com; rerun using the
same cache after transient failures.
