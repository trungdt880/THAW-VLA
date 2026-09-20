#!/bin/bash
# Single (ckpt, task_suite) eval job. Launches the policy server in the
# background, runs the LIBERO client to completion, then kills the server.
#
# Args:
#   $1 CKPT
#   $2 TASK_SUITE
#   $3 GPU_ID
#   $4 PORT
#   $5 STARVLA_DIR
#   $6 STARVLA_PYTHON
#   $7 LIBERO_HOME
#   $8 LIBERO_PYTHON
#   $9 NUM_TRIALS_PER_TASK
#   ${10} UNNORM_KEY
#   ${11} WANDB_PROJECT   (optional; enables eval->wandb logging)
#   ${12} WANDB_ENTITY    (optional)
#   ${13} WANDB_RUN_ID    (optional; = training run_id / shared wandb run to join)
#   ${14} WANDB_CKPT_STEP (optional; eval x-axis)
#   ${15} MAX_TASKS       (optional; -1 = all tasks, smoke: 1)

set -e

CKPT=$1
TASK_SUITE=$2
GPU_ID=$3
PORT=$4
STARVLA_DIR=$5
STARVLA_PYTHON=$6
LIBERO_HOME=$7
LIBERO_PYTHON=$8
NUM_TRIALS_PER_TASK=$9
UNNORM_KEY=${10}
WANDB_PROJECT=${11:-}
WANDB_ENTITY=${12:-}
WANDB_RUN_ID=${13:-}
WANDB_CKPT_STEP=${14:--1}
MAX_TASKS=${15:--1}

cd "${STARVLA_DIR}"

# Policy server must import THIS checkout's starVLA (distill modules: QwenGR00T.repa_head),
# not an editable-installed copy. Force it on PYTHONPATH for the server.
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# Output paths (mirrors auto_eval_libero.sh layout).
# Derive the run root the same way the loader does (share_tools.py: checkpoint.parents[1]),
# rather than splitting on a literal '/checkpoints/'. Training writes final_model/pytorch_model.pt,
# which has no such segment -- the old split returned the whole path, so logs/videos were mkdir'd
# *inside* the .pt file and the run died with 'Not a directory'.
model_root=$(dirname "$(dirname "$CKPT")")
folder_name=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${TASK_SUITE}/${folder_name}"
log_path="${model_root}/logs/${TASK_SUITE}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"

###################################################################
# 1. Policy server (starVLA env, GPU)
###################################################################
CUDA_VISIBLE_DEVICES=${GPU_ID} ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT}" \
    --port "${PORT}" \
    --use_bf16 \
    > "${log_path}/${folder_name}.server.log" 2>&1 &
server_pid=$!

# Wait for server readiness — poll until it accepts a tcp connect on PORT.
# (Avoids racing the client against a not-yet-listening server.)
for i in $(seq 1 60); do
    if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        echo "[ckpt=${folder_name} task=${TASK_SUITE}] server up on port ${PORT}"
        break
    fi
    sleep 5
done

###################################################################
# 2. LIBERO client (libero env, CPU torch)
###################################################################
export LIBERO_HOME
# LIBERO_CONFIG_PATH intentionally NOT set — defaults to ~/.libero which
# was pre-populated at install time. Setting it to ${LIBERO_HOME}/libero
# (the example default) points at a directory with no config.yaml, which
# makes libero prompt on stdin and crash the client.
export PYTHONPATH=${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# torch >= 2.6 defaults weights_only=True, which breaks LIBERO's torch.load of
# init-state files (numpy pickles). These are trusted local files -> force legacy load.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# silence torchvision video-deprecation UserWarning spam
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"

${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.max-tasks "${MAX_TASKS}" \
    --args.video-out-path "${video_out_path}" \
    --args.wandb-project "${WANDB_PROJECT}" \
    --args.wandb-entity "${WANDB_ENTITY}" \
    --args.wandb-run-id "${WANDB_RUN_ID}" \
    --args.wandb-ckpt-step "${WANDB_CKPT_STEP}" \
    2>&1 | tee "${log_path}/${folder_name}.client.log" || true

###################################################################
# 3. Tear down server
###################################################################
if kill -0 "${server_pid}" 2>/dev/null; then
    echo "[ckpt=${folder_name} task=${TASK_SUITE}] killing server pid=${server_pid}"
    kill "${server_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${server_pid}" 2>/dev/null || true
fi

echo "[ckpt=${folder_name} task=${TASK_SUITE}] done. Videos: ${video_out_path}, Log: ${log_path}/${folder_name}.client.log"
