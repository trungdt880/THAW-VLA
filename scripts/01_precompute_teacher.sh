#!/usr/bin/env bash
# Extract Cosmos3-Nano teacher features into an offline cache.
# This runs ONCE per dataset. Training then reads the cache; the teacher is never
# loaded during training and never at inference.
#
#   ./scripts/01_precompute_teacher.sh libero      # 2 cams -> target_dim 2*4096
#   ./scripts/01_precompute_teacher.sh gr1         # 1 cam  -> target_dim 1*4096
#
# Env: NGPU (default 4), TAP (24), MODE (pooled), BS (16), TEACHER_PY, OUT
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
TASK=${1:?"usage: $0 libero|gr1"}

case "$TASK" in
  libero) CFG=configs/libero_distill_qwen35_0p8b.yaml; NCAM=2; DEF_OUT=playground/caches/libero_cosmos3nano ;;
  gr1)    CFG=configs/gr1_distill_qwen35_0p8b.yaml;    NCAM=1; DEF_OUT=playground/caches/gr1_cosmos3nano ;;
  *) echo "unknown task: $TASK (want libero|gr1)"; exit 1 ;;
esac

PY=${TEACHER_PY:-$REPO/.venv_teacher/bin/python}
OUT=${OUT:-$DEF_OUT}
TAP=${TAP:-24}; MODE=${MODE:-pooled}; BS=${BS:-16}; LIMIT=${LIMIT:-0}
# GPU_LIST selects which physical GPUs the shards use; NGPU is derived from it. Previously the
# device was hardcoded to the shard rank, so shards always landed on 0..NGPU-1 with no way to
# avoid GPUs that were already busy.
GPU_LIST=${GPU_LIST:-$(seq -s, 0 $(( ${NGPU:-4} - 1 )))}
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NGPU=${#GPUS[@]}
[ -x "$PY" ] || { echo "teacher venv missing: run ./scripts/00_setup_env.sh --teacher"; exit 1; }

export PYTHONPATH=$REPO NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
NT=${NTHREADS:-16}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
mkdir -p "$OUT" logs

echo "[precompute] task=$TASK n_cam=$NCAM tap=$TAP mode=$MODE out=$OUT shards=$NGPU gpus=$GPU_LIST"
pids=(); ranks=()
for RANK in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$RANK]} FASTWAM_SHARD_RANK=$RANK FASTWAM_SHARD_WORLD=$NGPU \
    "$PY" tools/cosmos3_precompute_targets.py \
      --config_yaml "$CFG" --out_cache_root "$OUT" \
      --variant cosmos3nano --tap_layer "$TAP" --mode "$MODE" \
      --n_cam "$NCAM" --limit "$LIMIT" --batch_size "$BS" \
      > "logs/precompute_${TASK}_rank${RANK}.log" 2>&1 &
  pids+=($!); ranks+=($RANK); echo "  rank $RANK -> GPU ${GPUS[$RANK]} (pid ${pids[-1]})"
done

# Bare `wait` always returns 0, so a sweep where every shard crashed used to report success.
# Wait per-pid and keep the exit codes.
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[precompute] RANK ${ranks[$i]} FAILED -- see logs/precompute_${TASK}_rank${ranks[$i]}.log" >&2
    failed=$((failed+1))
  fi
done
if [ "$failed" -gt 0 ]; then
  echo "[precompute] FAILED: $failed/$NGPU shards crashed. Cache at $OUT is incomplete -- do not train on it." >&2
  exit 1
fi

# A shard can exit 0 and still have written nothing; require the cache to exist.
if ! find "$OUT" -name 'meta.json' -print -quit | grep -q .; then
  echo "[precompute] FAILED: no meta.json under $OUT -- nothing was written." >&2
  exit 1
fi

echo "[precompute] done. Cache: $OUT"
echo "[precompute] valid-fraction per dataset:"
find "$OUT" -name meta.json | while read -r m; do
  python3 - "$m" <<'EOF'
import json,sys,os
m=sys.argv[1]; d=json.load(open(m))
print(f"    {os.path.basename(os.path.dirname(m)):46s} n_samples={d.get('n_samples')} partial={d.get('partial', False)}")
EOF
done
