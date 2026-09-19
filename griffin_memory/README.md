# Griffin-inspired language model with three-tier memory

A self-contained, **trainable from-scratch reference implementation** of the
uploaded “Production-Grade Long-Form Creative-Writing Language Model” diagram.
It does not import QALF, load pretrained model weights, download a tokenizer, or
require a hosted retrieval service. Existing `qalf/` code is unchanged.

All illustrated components are implemented, including the retrieval write/read
path during **both training and generation**. The diagram does not specify an
executable mathematical model; the choices below make that model explicit.
This is **not** an exact reproduction of Google's Griffin/RecurrentGemma, nor a
claim of production-scale performance or guaranteed long-form coherence.

## Single DGX Spark and starter datasets

See **[PG19_RUNBOOK.md](PG19_RUNBOOK.md)** for the end-to-end PG-19 workflow
(`scripts/pg19_full_run.sh`: prepare → plan → pilot → train → samples → evaluate
→ report), budget arithmetic, expected wall-clock time and troubleshooting.
See **[DGX_SPARK.md](DGX_SPARK.md)** for TinyStories / PG-19 recommendations,
source-aware download/preparation commands, and single-Spark BF16 launch profiles.
`prepare_corpus.py` preserves official splits, keeps books whole, deduplicates
against held-out data, and bounds BPE fitting. It downloads no model weights.

## Quick start (from the repository root)

Requires Python **3.11+** and PyTorch **2.5+**. Use the appropriate PyTorch wheel
for your CPU/CUDA platform; the requirements do not select a CUDA driver.

```bash
python -m venv griffin_memory/.venv
source griffin_memory/.venv/bin/activate
pip install -r griffin_memory/requirements.txt

# Small, original example corpus; no dataset downloads.
python -m griffin_memory.prepare \
  --input griffin_memory/examples/stories.jsonl \
  --out griffin_memory/artifacts/demo-data --val-fraction 0.2

# All model and memory components enabled, including ANN retrieval.
python -m griffin_memory.train \
  --data griffin_memory/artifacts/demo-data \
  --config griffin_memory/configs/smoke.json \
  --out griffin_memory/artifacts/demo-run \
  --device cpu --threads 1 \
  --max-steps 80 --epochs 3 --warmup-steps 5 --learning-rate 0.002 \
  --eval-every 40 --save-every 40 --log-every 10 --eval-batches 0

python -m griffin_memory.generate \
  --checkpoint griffin_memory/artifacts/demo-run/last.pt \
  --prompt 'Mara found a blue lantern beneath the bridge.' \
  --max-new-tokens 128 --device cpu --threads 1 \
  --memory-out griffin_memory/artifacts/demo-run/memories.json

python -m griffin_memory.evaluate \
  --checkpoint griffin_memory/artifacts/demo-run/last.pt \
  --data griffin_memory/artifacts/demo-data --device cpu --threads 1

# Held-out generation report: prompts, greedy/sampled continuations, the real
# continuation, retrieval on/off continuation CE and memory diagnostics.
python -m griffin_memory.generate_samples \
  --checkpoint griffin_memory/artifacts/demo-run/best.pt \
  --data griffin_memory/artifacts/demo-data --split val \
  --out griffin_memory/artifacts/demo-run/samples \
  --documents 2 --prompt-tokens 48 --prefill-tokens 96 \
  --max-new-tokens 64 --continuation-tokens 64 --device cpu --threads 1

python -m unittest discover -s griffin_memory/tests -v
```

The smoke model has only **39,938 parameters**. This checks mechanics, not
creative-writing quality. A few training steps will produce poor text. The demo
corpus is deliberately tiny and must not be mistaken for a pretraining dataset.
Use a new output directory when repeating preparation or starting a fresh run.
`artifacts/`, virtual environments and Python caches are ignored by Git.

## Diagram → code

| Illustrated component | Actual implementation |
|---|---|
| Token embeddings and logits | Learned embedding and tied output projection, randomly initialized |
| Griffin-style decoder layers | Pre-norm residual hybrid layers, each containing recurrence **and** local attention |
| RMSNorm | FP32 RMS statistic, learned scale, configurable epsilon |
| Temporal Conv1D | Causal depthwise convolution; per-layer `kernel_size - 1` activation cache |
| RG-LRU state | Input-dependent, stable diagonal recurrence; persistent FP32 state per layer |
| GeLU branch | Independent projection + GeLU, multiplied with the recurrent output |
| Sliding-window multi-query attention | Many query heads, **one shared K/V head**, causal distance mask |
| RoPE | Rotary Q/K encoding using absolute stream positions across chunks |
| Local KV cache / recent memory | Last `window_size - 1` rotated keys and unrotated values per layer |
| Gated MLP | GeGLU feed-forward sublayer with residual connection |
| Semantic memory | All per-layer recurrent vectors carried forward across a document |
| Long-range ANN index | Multi-table random-hyperplane LSH, Hamming-radius-one probing, cosine reranking |
| Chapter/scene chunks | Completed fixed-token fragments of the source story, exact token IDs and source metadata |
| Key/value embeddings + metadata | Learned key and value projections of contextual chunk summaries; document ID and token span |
| Write/retrieve top-K | Causal post-chunk writes, token-specific top-K reads, learned gated residual fusion |

Files: `model.py` (decoder and streaming state), `memory.py` (ANN and retrieval
adapter), `prepare.py` / `prepare_corpus.py` / `tokenizer.py` / `data.py` (corpus
pipeline), `train.py`, `evaluate.py`, `generate.py`, `generate_samples.py`
(batch held-out generation reports), `run_report.py` (budget planning, throughput
estimates, `run_report.md`), `scripts/train_spark.sh`,
`scripts/pg19_full_run.sh` (end-to-end driver), `configs/`, `examples/`, and
`tests/`. See [PG19_RUNBOOK.md](PG19_RUNBOOK.md).

## Mathematical and architectural choices

### Decoder layout

For every layer:

```text
x ← x + recurrent(RMSNorm(x))
x ← x + local_MQA_with_RoPE(RMSNorm(x))
if this is the last layer:
    x ← x + retrieval_adapter(RMSNorm(x), earlier_chunk_bank)
x ← x + GeGLU_MLP(RMSNorm(x))
```

Final RMSNorm and a tied vocabulary projection produce next-token logits.
The diagram's ambiguous arrows are interpreted as residual/state connections,
not cycles in the feed-forward training graph. Both temporal sublayers occur in
every layer, rather than alternating Griffin's recurrent and attention layers.
There is a single shared retrieval adapter before the last layer's MLP, avoiding
one separate ANN search at every layer. No dropout is used.

### Recurrent block

For normalized layer input `x_t`, first compute a causal depthwise convolution
`u_t = Conv1D(W_u x_t)`. Gates are dense learned projections (not Griffin's
block-diagonal implementation). With elementwise operations:

```text
i_t = sigmoid(W_i u_t + b_i)
r_t = sigmoid(W_r u_t + b_r)
log(a_t) = -8 · softplus(lambda) · r_t
a_t = exp(log(a_t))
h_t = a_t · h_(t-1) + sqrt(1 - a_t²) · (i_t · u_t)
y_t = W_o [h_t · GELU(W_g x_t + b_g)]
```

`h_0 = 0`. The input multiplier is used at all positions, including the first;
there is no special first-token variance override. Initial ungated decay ranges
from 0.9 to 0.999. `-expm1(2 * log(a_t))` and a small positive floor stabilize the
square root. Recurrence and normalization statistics stay FP32 under AMP.

### Attention

RoPE rotates query/key pairs with configurable base `rope_theta`. A query at
position `p` can attend to keys `q` only when `0 <= p-q < window_size`, including
itself. Caches keep one K/V head and are trimmed after each call. PyTorch SDPA
handles attention, but this reference constructs a dense mask for the current
chunk plus its local cache; it is not a custom sparse-attention kernel.

### Long-range memory and trainability

A completed chunk's final contextual hidden states are mean-pooled over valid
input tokens only, detached, and stored alongside their **original token IDs**.
Each entry also contains key/value embeddings, source metadata, document ID,
and an exclusive `[start, end)` token span (including BOS in stream numbering).
FIFO eviction bounds storage to `memory_capacity` entries **per document lane**.
Writes do not include the shifted target beyond the input chunk.

At each token, the adapter projects a query. LSH selects up to `memory_top_k`
earlier chunks, cosine scores are temperature-scaled and softmaxed, and their
learned values are mixed. A sigmoid gate and output projection add the retrieved
information as a residual. The gate starts with bias -2, not a fixed zero gate.
The query, selected keys, values, temperature, gate, and fusion projection all
learn through next-token cross-entropy. Candidate selection and storage writes
are non-differentiable; stored summaries are detached, but selected key/value
projections are recomputed with gradients. No external embedding model or
separately pretrained retriever is needed.

The index is rebuilt from current projected keys on each read call, preventing
stale projection weights after optimizer updates. If probing finds fewer than K
candidates, retrieval falls back to a full scan. Set `memory_backend` to `exact`
for a full cosine search baseline; `lsh` is the default and is a real approximate
index, not a renamed dense lookup. With small stores, fallback may be common.

**“Exact recall” caveat:** the original token chunks are preserved exactly while
resident in the store and can be inspected/exported. The decoder reads compressed
learned values, not the full retrieved text through a separate cross-attention
encoder or copy head. Neither ANN selection nor neural reconstruction guarantees
verbatim recall. Chapter/scene content is split at fixed token boundaries, not
by an automatic semantic scene detector. This is a deliberate deterministic,
streaming-compatible interpretation of the diagram's chunk store.

### Causality, boundaries and gradients

- A chunk can retrieve only chunks completed **before that chunk started**.
  Its own summary is written after its logits have been computed.
- Fixed memory boundaries apply equally to teacher forcing and generation.
  `forward()` refuses calls crossing a boundary; `generate.prefill()` splits long
  prompts correctly. Partial calls accumulate a summary until the boundary.
- Every training lane holds one document. Documents are shuffled; chunks within
  them are never shuffled. A batch group is not refilled until its longest
  document ends; shorter lanes receive ignored, right-padded targets.
- All recurrent states, convolution caches, KV caches, pending summaries, and
  retrieval stores are reset between document groups and before validation.
  There is no cross-document or train/validation retrieval.
- Truncated backpropagation detaches carried states after each chunk. The
  forward history persists, but gradients do not extend into previous chunks.
  Past summaries are therefore not re-encoded/backpropagated through when read.

## Prepare your corpus

Preferred format: one **whole story/book/document per JSONL record**:

```json
{"id":"story-001","text":"Chapter One\n...\n\nChapter Two\n...","metadata":{"title":"Example","genre":"fantasy"}}
```

`text` is required unless both `prompt` and `reply` are provided (supported for
this repository's existing data). `id` and `metadata` are optional. Keep scenes
that need to share memory inside the same document; separate records deliberately
cannot retrieve from each other. Arbitrary chapter/scene labels may be carried
in source metadata, but the memory chunk boundaries remain token-count based.
Plain UTF-8 files are also accepted: each file is one document unless you specify
`--document-separator '<|endoftext|>'` (useful for TinyStories text files).

The default tokenizer is a reversible UTF-8 byte vocabulary: PAD=0, BOS=1, EOS=2,
bytes=3..258. This has no unknown-token problem, but uses many tokens per word.
For real training, train a byte-level BPE tokenizer **on the training split only**:

```bash
pip install 'tokenizers>=0.20,<1'
python -m griffin_memory.prepare \
  --input /path/to/stories.jsonl /path/to/more-stories.jsonl \
  --out griffin_memory/artifacts/corpus \
  --tokenizer bpe --vocab-size 16000 --val-fraction 0.01 --seed 42
```

Preparation spools text to disk, splits by a seeded hash of document content,
and writes little-endian uint32 token streams, int64 offset arrays, metadata,
tokenizer, and a SHA-256 manifest. Exact duplicate text is assigned to the same
split, but is not deduplicated. Near-duplicates require external deduplication.
An empty requested split is an error; for a training-only corpus explicitly use
`--val-fraction 0`. A single input text file without separators is only one
document, so cannot produce both splits. The tokenizer's actual vocabulary size
is automatically applied to the model before initialization.

Training memory-maps tokens, loading only active chunks as tensors. Offset arrays
are memory-mapped; metadata is loaded in RAM. Preparation currently loads one
plain-text file at a time; prefer JSONL for very large corpora. Dataset hashes
are verified once at training/evaluation startup. BPE training is local and does
not contact Hugging Face, despite using its open-source `tokenizers` library.

## Training a larger model

Parameter counts below use vocabulary size 16,000 except for the smoke model:

| Config | Layers × width | Parameters | Role |
|---|---:|---:|---|
| `smoke.json` | 2 × 32 | 39,938 (259-token vocabulary) | CPU tests |
| `small.json` | 6 × 256 | 10,670,338 | Small single-device experiments |
| `medium.json` | 12 × 768 | 128,212,226 | Larger experiments; profile on your hardware |

No GPU model was specified; these are architecture presets, **not guarantees of
VRAM fit or throughput**. Example for a CUDA GPU supporting BF16:

```bash
python -m griffin_memory.train \
  --data griffin_memory/artifacts/corpus \
  --config griffin_memory/configs/small.json \
  --out griffin_memory/artifacts/small-run \
  --device cuda --precision bf16 \
  --batch-size 4 --accumulation-steps 8 \
  --max-steps 20000 --epochs 10 --warmup-steps 500 \
  --learning-rate 0.0003 --weight-decay 0.1 --clip-grad 1.0 \
  --eval-every 250 --eval-batches 64 --save-every 250 --log-every 10
```

Use FP16 plus automatic loss scaling on GPUs without BF16; FP32 works on CPU or
CUDA. `--device auto` selects CUDA if available; it accepts `cpu`, `cuda`, or
`cuda:N` and rejects unsupported strings, unavailable CUDA, or an out-of-range
GPU index with a clear error before training starts. Reduce batch size first on OOM,
then model dimensions, chunk/window size, or memory capacity. Training ends at
**whichever comes first**: `max_steps` attempted updates or `epochs` full passes.
Make `epochs` large enough for your intended token budget.

AdamW uses `(beta1, beta2)=(0.9, 0.95)`, excludes 1-D parameters from decay, and
uses linear warmup plus cosine decay. Accumulation divides summed gradients by
the total number of non-ignored targets, not the number of chunks. Partial
accumulation at epoch-budget exhaustion is flushed. FP16 overflow skips are
logged; attempted updates still advance the schedule/data cursor.

`train.jsonl` records loss, LR, gradient norm, processed tokens, approximate
wall-clock token throughput, memory entries and validation results. Validation
is token-weighted next-token CE/perplexity; it does **not** measure narrative
coherence, retrieval recall, or writing quality. `--eval-batches 0` evaluates the
whole split; a positive cap evaluates only an initial prefix of its groups.
Perplexities from different tokenizers are not directly comparable.

## Checkpoint and resume

`last.pt` contains model/optimizer/scaler, scheduler inputs/update number, RNG
states, shuffled-document cursor, per-layer recurrent/convolution/KV state,
active retrieval stores and pending chunks. The tokenizer is embedded, so the
checkpoint can be copied alone for generation. `best.pt` is the best validation
checkpoint. Saves use an atomic rename and occur at optimizer boundaries.

```bash
python -m griffin_memory.train \
  --data griffin_memory/artifacts/corpus \
  --out griffin_memory/artifacts/small-run \
  --resume griffin_memory/artifacts/small-run/last.pt \
  --device cuda
```

The saved architecture and training hyperparameters are automatically restored;
conflicting explicit overrides or a changed dataset/tokenizer are rejected.
Logging, evaluation and save intervals may change. Use `--stop-after-steps N`
to stop cleanly for an interruption test without changing the planned LR
schedule. To continue after a completed `max_steps`/`epochs` budget, start a new
experiment; this strict resume interface intentionally does not silently alter
the original schedule. `--deterministic` requests deterministic PyTorch kernels.
Bitwise equality is tested for CPU FP32; it is not promised across hardware,
PyTorch releases, or different precision. Unclean shutdown loses progress since
the last completed save, not an in-progress partial gradient accumulation.
Only load checkpoints and datasets from trusted sources.

## Generation and inspection

`generate.py` handles arbitrarily long prompts by streaming bounded chunks.
It maintains semantic state, recent KV state, and an initially empty,
document-local retrieval bank, then generates token by token. It does **not**
reuse the active training document's bank from the checkpoint. Use `--prompt-file`
for long prompts, `--temperature 0` for greedy decoding, and `--top-k`/`--top-p`
for sampling. PAD/BOS cannot be generated; EOS terminates generation.
`--out` writes the full decoded prompt + continuation.

`--memory-out` exports resident original token chunks, decoded text, key/value
vectors, metadata, and pending token IDs to JSON. It is a diagnostic export, not
a cross-document knowledge base or a persisted chat-session resume interface.
Never commit exports/checkpoints containing private training text.

`generate_samples.py` is the batch counterpart used for run reports. It streams
an optional long prefill (`--prefill-tokens`) so the retrieval bank already holds
earlier parts of the same book, then writes `samples.md`/`samples.json` with the
real held-out prompt, greedy and sampled continuations, the real continuation,
teacher-forced continuation CE with the retrieval tier on and off
(`--no-ablation` skips the second pass) and memory diagnostics. `run_report.py
plan` converts corpus targets plus batch shape into steps and tokens,
`run_report.py estimate` extrapolates wall-clock time from a `train.jsonl`, and
`run_report.py build` assembles `run_report.md` from a run directory.
`scripts/pg19_full_run.sh` chains all of it with stage selection (`--only`,
`--skip`), a throughput pilot, `--dry-run`, and `--resume` for continuing. Its
generated and reference text carries whatever rights the source books have: do
not redistribute it.

## Validation performed and scaling limits

Tested on CPU with Python 3.11 / PyTorch 2.5.1:

- 58 tests pass, including optional locally trained BPE round trips, 12
  fixture-based corpus-preparation tests, and 20 tests covering the generation
  report, the run report (planning, throughput estimates, assembly) and the
  end-to-end driver script.
- Causality with populated retrieval memory; strict local attention windows.
- Chunked vs. tokenwise equivalence, RoPE positions and convolution/KV bounds.
- Finite, nonzero gradients for **every** trainable parameter with memory active.
- Bounded eviction, partial-chunk writes, padding exclusion and state round trips.
- Full-state resumed training is bitwise-identical to uninterrupted CPU training.
- Split integrity, data-cursor coverage, partial accumulation and eval isolation.
- Existing repository tests: all 11 pass.
- End-to-end byte/FP32 and BPE/CPU-BF16 training, evaluation and generation run.
- An 80-update smoke run reduced held-out byte-token loss to about 3.164 on the
  tiny bundled corpus. This is a learning sanity check, **not** a quality benchmark.
- A full `pg19_full_run.sh` smoke pass (prepare → plan → train → samples → eval →
  report) completes on CPU against a book-shaped fixture, and `--dry-run`,
  `--only`, `--pilot-steps` and `--resume` were exercised. Real PG-19 downloads
  and GPU paths remain unverified in this sandbox.

**CUDA/FP16 and the larger presets have not been exercised on a GPU here.**
The recurrence is a Python time-step scan, not a fused parallel scan; retrieval
uses a CPU LSH index and synchronizes with the model device. Every read refreshes
keys. Dense chunk-local attention, AdamW states and stored vectors also have
real memory costs. The trainer intentionally supports **one device**, not DDP,
FSDP, tensor parallelism, activation checkpointing, or a sharded retrieval service.
Multi-rank `torchrun` fails explicitly rather than racing on output files.

For production-scale pretraining, profile first, then replace the scan with a
validated fused implementation, batch/index retrieval on the accelerator, add
well-tested distributed document sharding and distributed stateful resume, and
benchmark long-range retrieval and story consistency. Architectural components
are present and trainable now; the diagram's “production-grade” label is not
itself evidence that these engineering and quality milestones have been met.
