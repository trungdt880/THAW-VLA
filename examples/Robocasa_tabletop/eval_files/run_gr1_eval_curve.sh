#!/usr/bin/env bash
# Sequentially eval a GR1 run's checkpoints (24 envs x N episodes each) on a chosen GPU set.
# Generalised from run_gr1_eval_baseline_curve.sh so it works for any run_id.
#
# Usage:
#   RUN_ID=starvla_distill_cosmos3nano_gr1_qwen35_0p8b_100k \
#   STEPS="100000 80000 60000 40000 20000" GPU_LIST=4,5,6,7 \
#   bash run_gr1_eval_curve.sh
#
# Each ckpt takes ~65-70 min (24 envs in parallel; wall-clock set by the slowest env, CPU/osmesa bound).
set -uo pipefail

RUN_ROOT=${RUN_ROOT:-/Data2/trungdt/playground_ckpts}
RUN_ID=${RUN_ID:?set RUN_ID}
CKPT_DIR="$RUN_ROOT/$RUN_ID/checkpoints"
EVAL_SH=${EVAL_SH:-$(dirname "${BASH_SOURCE[0]}")/run_gr1_eval_final.sh}
STEPS=${STEPS:?set STEPS, e.g. "100000 80000 60000 40000 20000"}

export GPU_LIST=${GPU_LIST:-4,5,6,7}
export BASE_PORT=${BASE_PORT:-6440}
export N_EPISODES=${N_EPISODES:-50}
export N_ENVS=${N_ENVS:-4}

echo "[curve] run_id=$RUN_ID gpus=$GPU_LIST n_ep=$N_EPISODES steps: $STEPS  $(date)"

for s in $STEPS; do
  CKPT="$CKPT_DIR/steps_${s}_pytorch_model.pt"
  if [ ! -f "$CKPT" ]; then echo "[curve] MISSING $CKPT -- skip"; continue; fi
  echo "[curve] ===== steps_${s} start $(date) ====="
  CKPT="$CKPT" bash "$EVAL_SH"
  echo "[curve] ===== steps_${s} done  $(date) ====="
  pkill -f "deployment/model_server/server_policy.py" 2>/dev/null
  pkill -f "eval_files/simulation_env.py" 2>/dev/null
  sleep 20
done

echo "[curve] ALL DONE $(date)"
