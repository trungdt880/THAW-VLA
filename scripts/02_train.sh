#!/usr/bin/env bash
# Train the student. Effective batch = per_device_batch_size * NGPU * GA.
#
#   ./scripts/02_train.sh configs/libero_distill_qwen35_0p8b.yaml
#   GPUS=0,1 ./scripts/02_train.sh configs/libero_baseline_qwen35_0p8b.yaml
#
# Env: GPUS (default 0,1), GA (1), RUN_ID (from yaml)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CFG=${1:?usage: $0 <config.yaml>}
[ -f "$CFG" ] || { echo "no such config: $CFG"; exit 1; }

GPUS=${GPUS:-0,1}
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
# Derive GA from the config; the trainer requires the launch flag to match the YAML.
GA=${GA:-$(grep -oE "^[[:space:]]*gradient_accumulation_steps:[[:space:]]*[0-9]+" "$CFG" | grep -oE "[0-9]+$" | head -1)}
GA=${GA:-1}

# NCCL: leave P2P ON. Disabling it forces ZeRO-2's per-step all-reduce off NVLink
# onto PCIe/host memory -- measured 3.8x slowdown on an NVSwitch box.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# wandb is OFF by default so a fresh clone runs without an account.
# To enable:  WANDB_MODE=online WANDB_ENTITY=<you> ./scripts/02_train.sh ...
export WANDB_MODE=${WANDB_MODE:-offline}
# (No WANDB_DIR: the trainer passes dir= to wandb.init explicitly, so runs always land in
#  playground/Checkpoints/<run_id>/wandb/ and the env var would have no effect.)
mkdir -p logs playground/Checkpoints

# Each concurrent run needs its own rendezvous port; the accelerate default (29500) collides.
MAIN_PORT=${MAIN_PORT:-29500}

echo "[train] cfg=$CFG gpus=$GPUS nproc=$NPROC ga=$GA port=$MAIN_PORT"

CUDA_VISIBLE_DEVICES=$GPUS .venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$NPROC" \
  --main_process_port "$MAIN_PORT" \
  --gradient_accumulation_steps "$GA" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CFG" \
  --trainer.gradient_accumulation_steps "$GA" \
  "${@:2}"
