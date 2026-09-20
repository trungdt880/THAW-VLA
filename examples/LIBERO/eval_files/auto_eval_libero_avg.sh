#!/usr/bin/env bash
# Full LIBERO eval (all 4 suites -> libero_avg) on EVERY checkpoint of a run, backlog + new.
# Processes one ckpt at a time (4 suites in parallel, one GPU each), so eval shares the box
# with training without thrashing. Per-suite success logs to wandb (via eval_libero.py); this
# driver also prints "[libero_avg] step=<N> avg=<A> | <per-suite>" for the trajectory.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STARVLA_DIR=$(pwd)

RUN_ROOT_DIR=${RUN_ROOT_DIR:-./playground/Checkpoints}
RUN_ID=${RUN_ID:?set RUN_ID}
CKPT_DIR="${RUN_ROOT_DIR}/${RUN_ID}/checkpoints"
DONE_FILE="${RUN_ROOT_DIR}/${RUN_ID}/.libero_avg_done"
touch "${DONE_FILE}"

STARVLA_PYTHON=${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-${STARVLA_DIR}/.venv_libero/bin/python}
LIBERO_HOME=${LIBERO_HOME:-/nobackup2/trungdt/code/LIBERO}
TASK_SUITES=(${TASK_SUITES:-libero_10 libero_goal libero_object libero_spatial})
GPU_LIST=(${GPU_LIST:-4 5 6 7})
BASE_PORT=${BASE_PORT:-6500}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-20}
MAX_TASKS=${MAX_TASKS:--1}
UNNORM_KEY=${UNNORM_KEY:-franka}
WANDB_PROJECT=${WANDB_PROJECT:-starVLA_Libero_distill}
WANDB_ENTITY=${WANDB_ENTITY:-tdao6-university-of-wisconsin-madison}
POLL_INTERVAL=${POLL_INTERVAL:-120}

echo "[libero_avg] run=${RUN_ID} ckpt_dir=${CKPT_DIR} suites=${TASK_SUITES[*]} trials=${NUM_TRIALS_PER_TASK}"

suite_rate() {  # $1 ckpt, $2 suite -> last "Total success rate: X" from that client log
  local ckpt="$1" suite="$2"
  local fn; fn=$(echo "$ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
  local cl="${RUN_ROOT_DIR}/${RUN_ID}/logs/${suite}/${fn}.client.log"
  grep -aoE "Total success rate: [0-9.]+" "$cl" 2>/dev/null | tail -1 | awk '{print $NF}'
}

eval_ckpt_full() {
  local ckpt="$1" step="$2"
  echo "=== [libero_avg] full eval ckpt step=${step} (suites: ${TASK_SUITES[*]}) ==="
  local pids=() job=0
  for suite in "${TASK_SUITES[@]}"; do
    local gpu=${GPU_LIST[$((job % ${#GPU_LIST[@]}))]}
    local port=$((BASE_PORT + job))
    bash ./examples/LIBERO/eval_files/eval_one.sh \
      "$ckpt" "$suite" "$gpu" "$port" \
      "$STARVLA_DIR" "$STARVLA_PYTHON" "$LIBERO_HOME" "$LIBERO_PYTHON" \
      "$NUM_TRIALS_PER_TASK" "$UNNORM_KEY" \
      "$WANDB_PROJECT" "$WANDB_ENTITY" "$RUN_ID" "$step" "$MAX_TASKS" &
    pids+=($!); job=$((job + 1)); sleep 5
  done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
  # collect per-suite rates -> libero_avg
  local sum=0 n=0 parts=""
  for suite in "${TASK_SUITES[@]}"; do
    local r; r=$(suite_rate "$ckpt" "$suite"); r=${r:-NA}
    parts="${parts} ${suite}=${r}"
    if [[ "$r" != "NA" ]]; then sum=$(awk "BEGIN{print $sum+$r}"); n=$((n+1)); fi
  done
  local avg="NA"; [[ "$n" -gt 0 ]] && avg=$(awk "BEGIN{printf \"%.4f\", $sum/$n}")
  echo "[libero_avg] step=${step} avg=${avg} |${parts}"
  # log libero_avg into the shared wandb run (best-effort)
  WANDB_DIR=/nobackup2/trungdt/.cache/wandb ${STARVLA_PYTHON} - "$RUN_ID" "$WANDB_PROJECT" "$WANDB_ENTITY" "$step" "$avg" <<'PY' 2>/dev/null || true
import sys, wandb
run_id, proj, ent, step, avg = sys.argv[1:6]
if avg != "NA":
    r = wandb.init(project=proj, entity=ent, id=run_id, resume="allow", settings=wandb.Settings(_disable_stats=True))
    r.log({"eval/libero_avg": float(avg), "eval/ckpt_step": int(step)})
    r.finish()
PY
}

while true; do
  mapfile -t ckpts < <(ls -1 "${CKPT_DIR}"/steps_*_pytorch_model.pt 2>/dev/null \
      | awk -F'steps_|_pytorch_model.pt' '{print $2" "$0}' | sort -n | cut -d' ' -f2-)
  did=0
  for ckpt in "${ckpts[@]}"; do
    step=$(echo "$ckpt" | sed -E 's/.*steps_([0-9]+).*/\1/')
    grep -qx "$step" "${DONE_FILE}" && continue
    eval_ckpt_full "$ckpt" "$step"
    echo "$step" >> "${DONE_FILE}"
    did=1
  done
  [[ "$did" -eq 0 ]] && { echo "[libero_avg] no new ckpt; sleeping ${POLL_INTERVAL}s"; sleep "${POLL_INTERVAL}"; }
done
