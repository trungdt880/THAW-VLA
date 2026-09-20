#!/usr/bin/env bash
# RoboCasa-GR1 eval: 24 environments, 50 episodes each (upstream protocol).
#
#   ./scripts/04_eval_gr1.sh playground/Checkpoints/<run>/checkpoints/steps_100000_pytorch_model.pt
#
# CPU-bound: 24 envs x n_envs rollouts of MuJoCo. Run it on an otherwise idle box --
# under load the policy servers can fail to bind and every env times out.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CKPT=${1:?usage: $0 <checkpoint.pt>}
export CKPT
export GPU_LIST=${GPU_LIST:-0,1,2,3}
export N_EPISODES=${N_EPISODES:-50}
export N_ENVS=${N_ENVS:-4}
export MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-720}
export N_ACTION_STEPS=${N_ACTION_STEPS:-12}
export BASE_PORT=${BASE_PORT:-6430}   # change to run two GR1 evals concurrently
export ROBOCASA_PY=${ROBOCASA_PY:-$REPO/.venv_robocasa/bin/python}
exec bash examples/Robocasa_tabletop/eval_files/run_gr1_eval_final.sh
