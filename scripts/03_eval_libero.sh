#!/usr/bin/env bash
# LIBERO eval: 4 suites in parallel, one GPU each.
#
#   ./scripts/03_eval_libero.sh playground/Checkpoints/<run>/checkpoints/steps_80000_pytorch_model.pt
#
# Env: GPUS ("0 1 2 3"), TRIALS (50 = standard protocol), LIBERO_HOME
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CKPT=${1:?usage: $0 <checkpoint.pt>}
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

read -ra G <<< "${GPUS:-0 1 2 3}"
TRIALS=${TRIALS:-50}          # 50/task = 500/suite = the protocol used by OpenVLA, pi0, StarVLA
LIBERO_HOME=${LIBERO_HOME:?set LIBERO_HOME to your LIBERO checkout}
LIBERO_PY=${LIBERO_PY:-$REPO/.venv_libero/bin/python}
# Override for a quick plumbing check, e.g. SUITES="libero_spatial" MAX_TASKS=1 TRIALS=2
read -ra SUITES <<< "${SUITES:-libero_spatial libero_object libero_goal libero_10}"
MAX_TASKS=${MAX_TASKS:--1}
BASE_PORT=${BASE_PORT:-7000}
PORTS=(); for _i in "${!SUITES[@]}"; do PORTS+=($((BASE_PORT + _i))); done

# eval_one.sh's readiness check is a bare TCP connect, so if another eval already holds these
# ports it succeeds against THAT server and this run scores the wrong checkpoint. Refuse to start.
for P in "${PORTS[@]}"; do
  if (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null; then
    exec 3<&- 3>&-
    echo "[eval-libero] ABORT: port $P already in use -- another eval is probably running." >&2
    echo "[eval-libero] Set BASE_PORT to a free range to run concurrently." >&2
    exit 1
  fi
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

echo "[eval-libero] ckpt=$(basename "$CKPT") trials/task=$TRIALS -> $((TRIALS*10)) episodes/suite"
pids=()
for i in "${!SUITES[@]}"; do
  bash ./examples/LIBERO/eval_files/eval_one.sh \
    "$CKPT" "${SUITES[$i]}" "${G[$((i % ${#G[@]}))]}" "${PORTS[$i]}" \
    "$REPO" "$REPO/.venv/bin/python" "$LIBERO_HOME" "$LIBERO_PY" \
    "$TRIALS" franka "" "" "" -1 "$MAX_TASKS" &
  pids+=($!); sleep 5
done
for p in "${pids[@]}"; do wait "$p" || true; done

RUN_ROOT=$(dirname "$(dirname "$CKPT")")
# Must match eval_one.sh's folder_name ($(NF-2)_$(NF-1)_$(NF)); don't hardcode 'checkpoints',
# or logs from a final_model/ checkpoint become unreadable.
FN="$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
# Aggregation must not run under `set -e`/`pipefail`: a suite whose log has no match makes
# grep exit 1, pipefail propagates it, and the script dies here -- printing no table and never
# reaching the completeness guard below, i.e. exactly when the guard matters most.
set +e +o pipefail
sum=0; n=0; NS=${#SUITES[@]}
for s in "${SUITES[@]}"; do
  f="$RUN_ROOT/logs/$s/$FN.client.log"
  # Only 'Total success rate' is the final number. 'Current total success rate' is a
  # running per-task value present even in crashed logs -- never read that one.
  r=$(grep -aoE "Total success rate: [0-9.]+" "$f" 2>/dev/null | tail -1 | awk '{print $NF}')
  e=$(grep -aoE "Total episodes: [0-9]+"      "$f" 2>/dev/null | tail -1 | awk '{print $NF}')
  printf "  %-15s %-7s (episodes %s)\n" "$s" "${r:-MISSING}" "${e:-MISSING}"
  if [ -n "$r" ]; then sum=$(awk "BEGIN{print $sum+$r}"); n=$((n+1)); fi
done
if [ "$n" -eq "$NS" ]; then
  echo "[eval-libero] AVG = $(awk "BEGIN{printf \"%.4f\", $sum/$NS}")"
else
  echo "[eval-libero] INCOMPLETE ($n/$NS suites) -- do not quote this run" >&2
  exit 1
fi
