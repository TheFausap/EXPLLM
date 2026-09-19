#!/usr/bin/env bash
# End-to-end experiment driver: prepare -> plan -> pilot -> train -> sample -> evaluate -> report.
#
# Example (DGX Spark, PG-19 starter corpus):
#   bash griffin_memory/scripts/pg19_full_run.sh runs/pg19-full \
#     --dataset pg19 --train-limit 1000 --val-limit 50 --include-test \
#     --token-budget 300000000 --samples 4 --prefill-tokens 4096
#
# Stage selection: --only prepare,plan   --skip pilot,sample
# Fresh experiments only: continuing one uses the trainer's own --resume interface
# (see griffin_memory/PG19_RUNBOOK.md); the driver never rewrites a schedule.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: pg19_full_run.sh RUN_DIRECTORY [options]

Corpus
  --data DIR                 Reuse an already prepared corpus (skips preparation)
  --dataset NAME             pg19 (default) or tinystories
  --train-jsonl FILE         Local documents instead of downloading; needs --val-jsonl
  --val-jsonl FILE           Local validation documents
  --test-jsonl FILE          Optional local test documents (never trained on)
  --train-limit N            Documents to sample (PG-19 preset default 1000; 0 = all)
  --val-limit N              Validation documents (PG-19 preset default 50; 0 = all)
  --include-test             Prepare the official PG-19 test books too
  --test-limit N             Test books to sample (0 = all)
  --tokenizer NAME           bpe (default) or byte
  --vocab-size N             BPE vocabulary size (default 16000)
  --tokenizer-max-characters N   BPE-fitting text budget (default 20000000; 0 = unlimited)
  --tokenizer-file FILE      Reuse an existing vocabulary
  --cache-dir DIR            Download cache (default griffin_memory/artifacts/downloads)
  --workers N                Concurrent PG-19 book downloads (default 4)

Model and training
  --config FILE              Model config (profile default: configs/small.json)
  --profile NAME             pg19 (default), tinystories or smoke; sets practical defaults
  --batch-size N             Documents per lane group
  --accumulation-steps N     Chunks per optimizer update
  --token-budget N           Planned prediction targets; converted into --max-steps
  --max-steps N              Optimizer updates
  --epochs N                 Epoch budget; whichever of steps/epochs is reached first wins
  --learning-rate X  --warmup-steps N  --weight-decay X  --clip-grad X  --min-lr-ratio X
  --seed N                   Data and initialization seed (default 42)
  --device NAME              auto, cpu, cuda or cuda:N (profile default)
  --precision NAME           fp32, bf16 or fp16 (profile default)
  --threads N                CPU intra-op threads
  --stop-after-steps N       Stop cleanly for an interruption test

Stages and reporting
  --pilot-steps N            Throughput probe updates before the real run (default 100; 0 disables)
  --eval-every N  --eval-batches N  --save-every N  --log-every N
  --eval-test                Evaluate the prepared test split after training
  --samples N                Documents to generate from (default 3)
  --sample-checkpoint NAME   best (default) or last
  --sample-select NAME       first (default), random, longest or shortest
  --sample-document-ids IDS  Explicit comma-separated document ids
  --prompt-tokens N          Prompt window (default 192)
  --prefill-tokens N         Same-document tokens streamed before the prompt (default 0)
  --start-fraction X         Window start position in the document, in [0, 1)
  --max-new-tokens N         Generated tokens per sample (default 256)
  --continuation-tokens N    Real held-out tokens scored as reference (default 256)
  --temperature X  --top-k N  --top-p X
  --no-sample-ablation       Skip the retrieval-off scoring pass
  --only STAGES              Run only these comma-separated stages
  --skip STAGES              Skip these comma-separated stages
  --dry-run                  Print commands instead of running them
  -h, --help                 Show this help
USAGE
}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}

run_dir=""
data=""
dataset="pg19"
train_jsonl=""
val_jsonl=""
test_jsonl=""
train_limit=""
val_limit=""
test_limit=0
include_test=0
tokenizer="bpe"
vocab_size=16000
tokenizer_max_characters=20000000
tokenizer_file=""
cache_dir="$repo/griffin_memory/artifacts/downloads"
workers=4
config=""
profile="pg19"
batch_size=""
accumulation_steps=""
token_budget=""
max_steps=""
epochs=""
learning_rate=""
warmup_steps=""
weight_decay=""
clip_grad=""
min_lr_ratio=""
seed=""
device=""
precision=""
threads=""
stop_after_steps=""
pilot_steps=""
eval_every=""
eval_batches=""
save_every=""
log_every=""
eval_test=0
samples=""
sample_checkpoint="best"
sample_select=""
sample_document_ids=""
prompt_tokens=""
prefill_tokens=""
start_fraction=""
max_new_tokens=""
continuation_tokens=""
temperature=""
top_k=""
top_p=""
sample_ablation=1
only=""
skip=""
resume=0
dry_run=0

# Values that came from the command line, passed through to the trainer untouched,
# so --resume never fights the checkpoint's saved schedule with a profile default.
train_overrides=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data) data=$2; shift 2 ;;
    --dataset) dataset=$2; shift 2 ;;
    --train-jsonl) train_jsonl=$2; shift 2 ;;
    --val-jsonl) val_jsonl=$2; shift 2 ;;
    --test-jsonl) test_jsonl=$2; shift 2 ;;
    --train-limit) train_limit=$2; shift 2 ;;
    --val-limit) val_limit=$2; shift 2 ;;
    --test-limit) test_limit=$2; shift 2 ;;
    --include-test) include_test=1; shift ;;
    --tokenizer) tokenizer=$2; shift 2 ;;
    --vocab-size) vocab_size=$2; shift 2 ;;
    --tokenizer-max-characters) tokenizer_max_characters=$2; shift 2 ;;
    --tokenizer-file) tokenizer_file=$2; shift 2 ;;
    --cache-dir) cache_dir=$2; shift 2 ;;
    --workers) workers=$2; shift 2 ;;
    --config) config=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --batch-size) batch_size=$2; train_overrides+=(--batch-size "$2"); shift 2 ;;
    --accumulation-steps) accumulation_steps=$2; train_overrides+=(--accumulation-steps "$2"); shift 2 ;;
    --token-budget) token_budget=$2; shift 2 ;;
    --max-steps) max_steps=$2; train_overrides+=(--max-steps "$2"); shift 2 ;;
    --epochs) epochs=$2; train_overrides+=(--epochs "$2"); shift 2 ;;
    --learning-rate) learning_rate=$2; train_overrides+=(--learning-rate "$2"); shift 2 ;;
    --warmup-steps) warmup_steps=$2; train_overrides+=(--warmup-steps "$2"); shift 2 ;;
    --weight-decay) weight_decay=$2; train_overrides+=(--weight-decay "$2"); shift 2 ;;
    --clip-grad) clip_grad=$2; train_overrides+=(--clip-grad "$2"); shift 2 ;;
    --min-lr-ratio) min_lr_ratio=$2; train_overrides+=(--min-lr-ratio "$2"); shift 2 ;;
    --seed) seed=$2; train_overrides+=(--seed "$2"); shift 2 ;;
    --device) device=$2; shift 2 ;;
    --precision) precision=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --stop-after-steps) stop_after_steps=$2; shift 2 ;;
    --pilot-steps) pilot_steps=$2; shift 2 ;;
    --eval-every) eval_every=$2; shift 2 ;;
    --eval-batches) eval_batches=$2; shift 2 ;;
    --save-every) save_every=$2; shift 2 ;;
    --log-every) log_every=$2; shift 2 ;;
    --eval-test) eval_test=1; shift ;;
    --samples) samples=$2; shift 2 ;;
    --sample-checkpoint) sample_checkpoint=$2; shift 2 ;;
    --sample-select) sample_select=$2; shift 2 ;;
    --sample-document-ids) sample_document_ids=$2; shift 2 ;;
    --prompt-tokens) prompt_tokens=$2; shift 2 ;;
    --prefill-tokens) prefill_tokens=$2; shift 2 ;;
    --start-fraction) start_fraction=$2; shift 2 ;;
    --max-new-tokens) max_new_tokens=$2; shift 2 ;;
    --continuation-tokens) continuation_tokens=$2; shift 2 ;;
    --temperature) temperature=$2; shift 2 ;;
    --top-k) top_k=$2; shift 2 ;;
    --top-p) top_p=$2; shift 2 ;;
    --no-sample-ablation) sample_ablation=0; shift ;;
    --only) only=$2; shift 2 ;;
    --skip) skip=$2; shift 2 ;;
    --resume) resume=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *) if [[ -n "$run_dir" ]]; then echo "Unexpected argument: $1" >&2; usage; exit 2; fi
       run_dir=$1; shift ;;
  esac
done

if [[ -z "$run_dir" ]]; then usage; exit 2; fi
run_dir=$(realpath -m "$run_dir")
case "$dataset" in
  pg19|tinystories) ;;
  *) echo "Unknown dataset: $dataset (expected pg19 or tinystories)" >&2; usage; exit 2 ;;
esac

case "$profile" in
  pg19)        profile_config=griffin_memory/configs/small.json; profile_batch=1; profile_accumulation=16
               profile_device=cuda; profile_precision=bf16; profile_steps=5000; profile_pilot=100 ;;
  tinystories) profile_config=griffin_memory/configs/small.json; profile_batch=4; profile_accumulation=4
               profile_device=cuda; profile_precision=bf16; profile_steps=5000; profile_pilot=100 ;;
  smoke)       profile_config=griffin_memory/configs/smoke.json; profile_batch=2; profile_accumulation=1
               profile_device=cpu; profile_precision=fp32; profile_steps=60; profile_pilot=0 ;;
  *) echo "Unknown profile: $profile (expected pg19, tinystories or smoke)" >&2; exit 2 ;;
esac
config=${config:-$profile_config}
device=${device:-$profile_device}
precision=${precision:-$profile_precision}
pilot_steps=${pilot_steps:-$profile_pilot}
epochs=${epochs:-3}
eval_every=${eval_every:-250}
save_every=${save_every:-100}
log_every=${log_every:-10}
samples=${samples:-3}
sample_select=${sample_select:-first}
prompt_tokens=${prompt_tokens:-192}
prefill_tokens=${prefill_tokens:-0}
max_new_tokens=${max_new_tokens:-256}
continuation_tokens=${continuation_tokens:-256}
temperature=${temperature:-0.8}
top_k=${top_k:-40}
top_p=${top_p:-0.95}

corpus_dir="$run_dir/corpus"
if [[ -n "$data" ]]; then corpus_dir=$(realpath -m "$data"); fi
train_dir="$run_dir/train"
driver_dir="$run_dir/driver"
samples_dir="$run_dir/samples"
eval_dir="$run_dir/eval"
checkpoint="$train_dir/best.pt"
if [[ "$sample_checkpoint" == "last" ]]; then checkpoint="$train_dir/last.pt"; fi

commands_log="$driver_dir/commands.log"
if (( dry_run )); then commands_log=/dev/null; else mkdir -p "$driver_dir"; : > "$commands_log"; fi

say() { printf '%s\n' "$*" >&2; }
run() {
  { printf '$'; printf ' %q' "$@"; printf '\n'; } >> "$commands_log"
  if (( dry_run )); then
    { printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'; } >&2
  else
    "$@"
  fi
}

stage_enabled() {
  local wanted=$1
  if [[ -n "$only" && ",$only," != *",$wanted,"* ]]; then return 1; fi
  if [[ -n "$skip" && ",$skip," == *",$wanted,"* ]]; then return 1; fi
  return 0
}

say "Run directory: $run_dir"
say "Profile: $profile · config $config · device $device · precision $precision"

# --------------------------------------------------------------------------- environment
if stage_enabled env && (( ! dry_run )); then
  say "== environment check =="
  "$PYTHON" - "$device" "$precision" <<'PY'
import platform
import sys

requested, precision = sys.argv[1], sys.argv[2]
try:
    import torch
except ImportError as error:
    raise SystemExit(f"PyTorch is unavailable: {error}. Install griffin_memory/requirements.txt in a "
                     "platform-correct environment; on DGX Spark keep the vendor PyTorch build.")
print({"python": platform.python_version(), "machine": platform.machine(), "torch": str(torch.__version__),
       "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
       "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})
if requested != "cpu" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but unavailable. Fix the driver/container, or run with "
                     "--device cpu --precision fp32. Do not swap in a generic CPU wheel to work around it.")
if precision == "fp16" and requested == "cpu":
    raise SystemExit("fp16 requires CUDA; use fp32 on CPU.")
if precision == "bf16" and requested != "cpu" and not torch.cuda.is_bf16_supported():
    raise SystemExit("bf16 is not supported on this GPU; use --precision fp16.")
target = torch.device("cpu") if requested == "cpu" else \
    (torch.device("cuda") if requested == "auto" and torch.cuda.is_available() else torch.device(requested))
# Exercise a real kernel and backward pass instead of trusting device enumeration.
x = torch.randn(64, 64, device=target, requires_grad=True)
y = (x @ x.T).float().square().mean()
y.backward()
if not torch.isfinite(y) or not bool(torch.isfinite(x.grad).all()):
    raise SystemExit("Matrix smoke check returned non-finite values.")
print({"matrix_smoke": float(y), "device_used": str(target), "finite_grad": True})
PY
fi

# --------------------------------------------------------------------------- prepare
if stage_enabled prepare && (( ! dry_run )); then
  say "== prepare corpus =="
  if [[ -f "$corpus_dir/manifest.json" ]]; then
    say "Reusing prepared corpus: $corpus_dir"
  elif [[ -n "$data" ]]; then
    echo "--data expects an existing prepared corpus, but $corpus_dir has no manifest.json" >&2
    echo "Drop --data to prepare a new corpus inside the run directory, or point --data at the right dataset." >&2
    exit 1
  elif [[ -n "$train_jsonl" ]]; then
    if [[ -z "$val_jsonl" ]]; then echo "--train-jsonl requires --val-jsonl" >&2; exit 2; fi
    prepare_args=(--input "$train_jsonl" --validation-input "$val_jsonl" --val-fraction 0
                  --out "$corpus_dir" --tokenizer "$tokenizer" --vocab-size "$vocab_size"
                  --tokenizer-max-characters "$tokenizer_max_characters")
    [[ -n "$test_jsonl" ]] && prepare_args+=(--test-input "$test_jsonl")
    [[ -n "$tokenizer_file" ]] && prepare_args+=(--tokenizer-file "$tokenizer_file")
    [[ -n "$seed" ]] && prepare_args+=(--seed "$seed")
    run "$PYTHON" -m griffin_memory.prepare "${prepare_args[@]}"
  else
    preset_args=()
    [[ -n "$train_limit" ]] && preset_args+=(--train-limit "$train_limit")
    [[ -n "$val_limit" ]] && preset_args+=(--val-limit "$val_limit")
    [[ -n "$tokenizer_file" ]] && preset_args+=(--tokenizer-file "$tokenizer_file")
    [[ -n "$seed" ]] && preset_args+=(--seed "$seed")
    (( include_test )) && preset_args+=(--include-test --test-limit "$test_limit")
    run "$PYTHON" -m griffin_memory.prepare_corpus --dataset "$dataset" --out "$corpus_dir" \
      --cache-dir "$cache_dir" --workers "$workers" --tokenizer "$tokenizer" --vocab-size "$vocab_size" \
      --tokenizer-max-characters "$tokenizer_max_characters" "${preset_args[@]}"
  fi
fi

if (( ! dry_run )) && [[ ! -f "$corpus_dir/manifest.json" ]]; then
  echo "No prepared corpus at $corpus_dir (the 'prepare' stage may have been skipped)" >&2
  exit 1
fi

# --------------------------------------------------------------------------- plan
plan_args=(--data "$corpus_dir" --config "$config" --epochs "$epochs"
           --batch-size "${batch_size:-$profile_batch}"
           --accumulation-steps "${accumulation_steps:-$profile_accumulation}")
if [[ -n "$token_budget" ]]; then plan_args+=(--token-budget "$token_budget")
elif [[ -n "$max_steps" ]]; then plan_args+=(--max-steps "$max_steps"); fi

if stage_enabled plan; then
  say "== plan =="
  run "$PYTHON" -m griffin_memory.run_report plan "${plan_args[@]}" --out "$driver_dir/plan.json"
  if (( ! dry_run )) && [[ -z "$max_steps" ]]; then
    max_steps=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["planned_steps"])' \
      "$driver_dir/plan.json")
  fi
  say "Planned updates: ${max_steps:-$profile_steps}"
fi
max_steps=${max_steps:-$profile_steps}

# --------------------------------------------------------------------------- pilot
if stage_enabled pilot && (( pilot_steps > 0 )); then
  say "== pilot throughput probe ($pilot_steps updates) =="
  pilot_args=(--data "$corpus_dir" --out "$driver_dir/pilot" --config "$config"
              --device "$device" --precision "$precision"
              --batch-size "${batch_size:-$profile_batch}"
              --accumulation-steps "${accumulation_steps:-$profile_accumulation}"
              --max-steps "$pilot_steps" --epochs "$epochs" --warmup-steps "${warmup_steps:-10}"
              --learning-rate "${learning_rate:-0.0003}" --eval-every 0 --save-every 0 --log-every 5
              --eval-batches 1)
  [[ -n "$threads" ]] && pilot_args+=(--threads "$threads")
  [[ -n "$stop_after_steps" ]] && pilot_args+=(--stop-after-steps "$stop_after_steps")
  rm -rf "$driver_dir/pilot"
  run "$PYTHON" -m griffin_memory.train "${pilot_args[@]}"
  if (( ! dry_run )); then
    run "$PYTHON" -m griffin_memory.run_report estimate --log "$driver_dir/pilot/train.jsonl" \
      --planned-steps "$max_steps" --out "$driver_dir/estimate.json"
    say "Compare the projected hours above with your budget before the full run starts."
  fi
fi

# --------------------------------------------------------------------------- train
if stage_enabled train; then
  say "== train =="
  train_args=(--data "$corpus_dir" --out "$train_dir" --device "$device")
  [[ -n "$threads" ]] && train_args+=(--threads "$threads")
  if (( resume )); then
    train_args+=(--resume "$train_dir/last.pt")
    if [[ ${#train_overrides[@]} -gt 0 ]]; then train_args+=("${train_overrides[@]}"); fi
    say "Resuming $train_dir/last.pt: saved hyperparameters are restored, conflicting overrides are rejected."
  else
    train_args+=(--config "$config" --precision "$precision"
                 --batch-size "${batch_size:-$profile_batch}"
                 --accumulation-steps "${accumulation_steps:-$profile_accumulation}"
                 --max-steps "$max_steps" --epochs "$epochs"
                 --warmup-steps "${warmup_steps:-100}" --learning-rate "${learning_rate:-0.0003}"
                 --weight-decay "${weight_decay:-0.1}" --clip-grad "${clip_grad:-1.0}"
                 --eval-every "$eval_every" --save-every "$save_every" --log-every "$log_every"
                 --eval-batches "${eval_batches:-64}")
    [[ -n "$min_lr_ratio" ]] && train_args+=(--min-lr-ratio "$min_lr_ratio")
    if [[ ${#train_overrides[@]} -gt 0 ]]; then train_args+=("${train_overrides[@]}"); fi
  fi
  [[ -n "$stop_after_steps" ]] && train_args+=(--stop-after-steps "$stop_after_steps")
  run "$PYTHON" -m griffin_memory.train "${train_args[@]}"
fi

# --------------------------------------------------------------------------- sample
if stage_enabled sample; then
  if (( dry_run )) || [[ -f "$checkpoint" ]]; then
    say "== generate samples =="
    sample_args=(--data "$corpus_dir" --out "$samples_dir" --device "$device" --precision "$precision"
                 --documents "$samples" --select "$sample_select" --prompt-tokens "$prompt_tokens"
                 --prefill-tokens "$prefill_tokens" --max-new-tokens "$max_new_tokens"
                 --continuation-tokens "$continuation_tokens" --temperature "$temperature"
                 --top-k "$top_k" --top-p "$top_p")
    [[ -n "$threads" ]] && sample_args+=(--threads "$threads")
    [[ -n "$sample_document_ids" ]] && sample_args+=(--document-ids "$sample_document_ids")
    [[ -n "$start_fraction" ]] && sample_args+=(--start-fraction "$start_fraction")
    (( sample_ablation )) || sample_args+=(--no-ablation)
    run "$PYTHON" -m griffin_memory.generate_samples --checkpoint "$checkpoint" "${sample_args[@]}"
  else
    say "Skipping samples: $checkpoint does not exist yet."
  fi
fi

# --------------------------------------------------------------------------- evaluate
if stage_enabled eval; then
  if (( dry_run )) || [[ -f "$train_dir/last.pt" ]]; then
    say "== evaluate =="
    eval_args=(--checkpoint "$train_dir/last.pt" --data "$corpus_dir" --device "$device"
               --precision "$precision" --batch-size "${batch_size:-$profile_batch}")
    [[ -n "$threads" ]] && eval_args+=(--threads "$threads")
    [[ -n "$eval_batches" ]] && eval_args+=(--max-batches "$eval_batches")
    run "$PYTHON" -m griffin_memory.evaluate "${eval_args[@]}" --split val --out "$eval_dir/val.json"
    if (( eval_test )); then
      run "$PYTHON" -m griffin_memory.evaluate "${eval_args[@]}" --split test --out "$eval_dir/test.json"
    fi
  else
    say "Skipping evaluation: $train_dir/last.pt does not exist yet."
  fi
fi

# --------------------------------------------------------------------------- report
if stage_enabled report && (( ! dry_run )); then
  say "== report =="
  report_args=(--run-dir "$run_dir" --data "$corpus_dir" --title "$(basename "$run_dir")")
  [[ -f "$samples_dir/samples.json" ]] && report_args+=(--samples "$samples_dir/samples.json")
  [[ -f "$eval_dir/val.json" ]] && report_args+=(--eval "$eval_dir/val.json")
  [[ -f "$eval_dir/test.json" ]] && report_args+=(--eval "$eval_dir/test.json")
  [[ -f "$driver_dir/plan.json" ]] && report_args+=(--plan "$driver_dir/plan.json")
  [[ -f "$driver_dir/estimate.json" ]] && report_args+=(--estimate "$driver_dir/estimate.json")
  "$PYTHON" -m griffin_memory.run_report build "${report_args[@]}" --out "$run_dir/run_report.md" > /dev/null
  say "Wrote $run_dir/run_report.md"
fi

say "Done. Artifacts: $run_dir"
