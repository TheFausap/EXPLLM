#!/usr/bin/env bash
# Fresh single-Spark pilot runs. For resume use python -m griffin_memory.train --resume ...
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 {tinystories|pg19} DATA_DIRECTORY RUN_DIRECTORY [extra train.py arguments]" >&2
  exit 2
fi
profile=$1
# Resolve user paths before changing to the repository root.
data=$(realpath "$2")
out=$(realpath -m "$3")
shift 3
case "$profile" in
  tinystories) batch=4; accumulation=4 ;;
  pg19) batch=1; accumulation=16 ;;
  *) echo "Unknown profile: $profile (expected tinystories or pg19)" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON=${PYTHON:-python}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
"$PYTHON" - <<'PY'
import platform
import torch
print({"machine": platform.machine(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda})
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Use a Spark-compatible NVIDIA PyTorch environment; do not install a generic CPU wheel.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16 unavailable. Check the Spark driver/container/PyTorch installation.")
print({"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()})
# Exercise both a kernel and backward rather than trusting device enumeration alone.
x = torch.randn(64, 64, device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = (x @ x.T).float().square().mean()
y.backward()
torch.cuda.synchronize()
if not torch.isfinite(y) or not torch.isfinite(x.grad).all():
    raise SystemExit("CUDA/BF16 smoke check returned non-finite values.")
print("CUDA/BF16 forward + backward: OK")
PY
exec "$PYTHON" -m griffin_memory.train \
  --data "$data" --out "$out" --config griffin_memory/configs/small.json \
  --device cuda --precision bf16 --threads 4 \
  --batch-size "$batch" --accumulation-steps "$accumulation" \
  --max-steps 5000 --epochs 3 --warmup-steps 100 \
  --learning-rate 0.0003 --weight-decay 0.1 --clip-grad 1.0 \
  --eval-every 250 --eval-batches 64 --save-every 100 --log-every 10 \
  "$@"
