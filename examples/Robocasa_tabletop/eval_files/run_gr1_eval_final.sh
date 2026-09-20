#!/usr/bin/env bash
# GR1 final-ckpt eval: N policy servers (GPUs 0-3 free mem) + 24 sim envs dispatched round-robin,
# each with n_envs parallel rollouts (osmesa CPU render across 128 cores).
# Fixes baked in: osmesa (no MUJOCO_EGL_DEVICE_ID), NUMBA_DISABLE_JIT (placement_samplers segfault), av installed.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$REPO"

STARVLA_PY=${STARVLA_PY:-$REPO/.venv/bin/python}
ROBOCASA_PY=${ROBOCASA_PY:-/Data2/trungdt/robocasa_eval/.venv/bin/python}
CKPT=${CKPT:?set CKPT to the .pt checkpoint}
GPU_LIST=${GPU_LIST:-0,1,2,3}          # policy-server GPUs (use train's free headroom)
N_EPISODES=${N_EPISODES:-50}
N_ENVS=${N_ENVS:-4}                     # parallel rollouts per sim client
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
N_ACTION_STEPS=${N_ACTION_STEPS:-12}
BASE_PORT=${BASE_PORT:-6430}

export PYTHONPATH=$REPO:${PYTHONPATH:-}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
unset MUJOCO_EGL_DEVICE_ID
export NUMBA_DISABLE_JIT=1
export TOKENIZERS_PARALLELISM=false

# --- THREAD CAPS (required) ---------------------------------------------------------------
# Without these every policy server spawns ~one OMP thread per core (259 threads observed on a
# 128-core box) and 24 sim clients do the same. Load hit 117/128 and servers hung in
# futex_wait_queue_me after loading weights but BEFORE binding their socket -- so all 24 envs
# died with "TimeoutError: Failed to connect to server within 300 seconds" and the sweep
# reported NA across the board. Same class of failure as the vjepa2 multi-GPU model-build hang.
SERVER_THREADS=${SERVER_THREADS:-8}
SIM_THREADS=${SIM_THREADS:-2}

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NG=${#GPUS[@]}
SAVE_ROOT=$(dirname "$(dirname "$CKPT")")
ckpt_name=$(basename "$CKPT" .pt)
LOG_DIR="${SAVE_ROOT}/eval_final_${ckpt_name}/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOG_DIR"
echo "[eval-final] ckpt=$ckpt_name gpus=$GPU_LIST n_env=$N_ENVS n_ep=$N_EPISODES logs=$LOG_DIR  $(date)"

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

# --- Step 0: refuse to start if our ports are already taken -------------------------------
# The readiness probe below is a bare TCP connect: if another eval is already listening on these
# ports, it succeeds against THAT server and this sweep silently scores the wrong checkpoint.
for i in $(seq 0 $((NG-1))); do
  P=$((BASE_PORT + i))
  if (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null; then
    exec 3<&- 3>&-
    echo "[eval-final] ABORT: port $P is already in use -- another eval is probably running." >&2
    echo "[eval-final] Set BASE_PORT to a free range to run concurrently." >&2
    exit 1
  fi
done

# --- Step 1: one policy server per GPU ---
SERVER_PIDS=()
for i in $(seq 0 $((NG-1))); do
  G=${GPUS[$i]}; PORT=$((BASE_PORT + i))
  echo "[eval-final] server GPU$G port$PORT"
  CUDA_VISIBLE_DEVICES=$G \
  OMP_NUM_THREADS=$SERVER_THREADS MKL_NUM_THREADS=$SERVER_THREADS \
  OPENBLAS_NUM_THREADS=$SERVER_THREADS NUMEXPR_NUM_THREADS=$SERVER_THREADS \
  "$STARVLA_PY" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$PORT" --use_bf16 --idle_timeout -1 \
    > "$LOG_DIR/server_gpu${G}_port${PORT}.log" 2>&1 &
  SERVER_PIDS+=($!)
  sleep 8
done

# --- Wait for servers to ACTUALLY accept connections, not a blind sleep ---------------------
# The old `sleep 40` assumed the servers were up. Under load they can take far longer to bind,
# and the clients then burn their own 300s timeout and the whole sweep reports NA. Poll the
# ports instead and fail loudly if they never come up.
READY_TIMEOUT=${READY_TIMEOUT:-900}
echo "[eval-final] waiting up to ${READY_TIMEOUT}s for $NG servers to accept connections"
deadline=$(( SECONDS + READY_TIMEOUT ))
while :; do
  ready=0
  for i in $(seq 0 $((NG-1))); do
    (exec 3<>/dev/tcp/127.0.0.1/$((BASE_PORT + i))) 2>/dev/null && { ready=$((ready+1)); exec 3<&- 3>&-; }
  done
  [ "$ready" -ge "$NG" ] && { echo "[eval-final] all $NG servers ready after $((SECONDS))s"; break; }
  if [ $SECONDS -ge $deadline ]; then
    echo "[eval-final] ERROR: only $ready/$NG servers bound after ${READY_TIMEOUT}s — aborting this ckpt"
    for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null; done
    exit 1
  fi
  sleep 5
done

# --- Step 2: dispatch 24 envs round-robin over servers (all parallel) ---
COUNT=0
for ENV_NAME in "${ENV_NAMES[@]}"; do
  i=$((COUNT % NG)); G=${GPUS[$i]}; PORT=$((BASE_PORT + i))
  VOUT="${LOG_DIR}/videos/${ENV_NAME//\//_}"; mkdir -p "$VOUT"
  echo "[eval-final] env $ENV_NAME -> GPU$G port$PORT"
  CUDA_VISIBLE_DEVICES=$G \
  OMP_NUM_THREADS=$SIM_THREADS MKL_NUM_THREADS=$SIM_THREADS \
  OPENBLAS_NUM_THREADS=$SIM_THREADS NUMEXPR_NUM_THREADS=$SIM_THREADS \
  "$ROBOCASA_PY" examples/Robocasa_tabletop/eval_files/simulation_env.py \
    --args.env_name "$ENV_NAME" --args.port "$PORT" \
    --args.n_episodes "$N_EPISODES" --args.n_envs "$N_ENVS" \
    --args.max_episode_steps "$MAX_EPISODE_STEPS" --args.n_action_steps "$N_ACTION_STEPS" \
    --args.video_out_path "$VOUT" --args.pretrained_path "$CKPT" \
    > "$LOG_DIR/eval_${ENV_NAME//\//_}.log" 2>&1 &
  COUNT=$((COUNT+1))
  sleep 3
done

echo "[eval-final] all ${#ENV_NAMES[@]} envs dispatched; waiting for completion"
while pgrep -f "eval_files/simulation_env.py" > /dev/null; do sleep 30; done

echo "[eval-final] all sims done; shutting servers"
for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null; done

# --- Step 3: aggregate SR from per-env logs ---
echo "[eval-final] === PER-ENV SUCCESS RATES ==="
tot=0; nsum=0
for ENV_NAME in "${ENV_NAMES[@]}"; do
  L="$LOG_DIR/eval_${ENV_NAME//\//_}.log"
  sr=$(grep -oE "Success rate: [0-9.]+" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  printf "  %-70s %s\n" "$(basename $ENV_NAME)" "${sr:-NA}"
  if [ -n "$sr" ]; then tot=$(echo "$tot + $sr"|bc -l); nsum=$((nsum+1)); fi
done
NENV=${#ENV_NAMES[@]}
if [ "$nsum" -eq "$NENV" ]; then
  echo "[eval-final] AVG SR over $nsum/$NENV envs: $(echo "scale=4; $tot/$nsum"|bc -l)"
else
  echo "[eval-final] INCOMPLETE SWEEP: only $nsum/$NENV envs reported -- refusing to print an average." >&2
  echo "[eval-final] A partial mean is not comparable to a full one; rerun the missing envs." >&2
fi
echo "[eval-final] DONE $(date) | logs: $LOG_DIR"
