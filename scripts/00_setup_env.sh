#!/usr/bin/env bash
# Create the environments with uv.
#
#   ./scripts/00_setup_env.sh            # main training/eval env  -> .venv
#   ./scripts/00_setup_env.sh --teacher  # Cosmos3-Nano extraction -> .venv_teacher
#   ./scripts/00_setup_env.sh --libero   # LIBERO simulator        -> .venv_libero
#   ./scripts/00_setup_env.sh --robocasa # RoboCasa-GR1 simulator  -> .venv_robocasa
#
# Four environments are needed because their transformers pins conflict:
#   .venv          transformers 5.9  (Qwen3.5 needs 5.x + flash-linear-attention)
#   .venv_teacher  transformers 5.12 (Cosmos3-Nano tower)
#   .venv_libero   CPU torch + robosuite/mujoco; never imports the policy
#   .venv_robocasa robosuite 1.5 + robocasa-gr1-tabletop-tasks (LIBERO needs robosuite 1.4,
#                  so the two simulators cannot share an environment)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

case "${1:-}" in
  --teacher)
    # Cosmos3-Nano provides Cosmos3OmniForConditionalGeneration, which needs transformers >=5.12
    # plus diffusers and numpy 2.x. That combination is incompatible with the student env's pins,
    # which is exactly why this is a separate venv.
    echo "[setup] .venv_teacher (Cosmos3-Nano feature extraction)"
    uv venv --python 3.10 .venv_teacher
    VIRTUAL_ENV=$REPO/.venv_teacher uv pip install \
      torch==2.7.1 "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
    # torchvision MUST be pinned: the dataloader's video path uses torchvision.io.VideoReader,
    # which was removed after 0.22.x. An unpinned install resolves to 0.26 and every shard dies
    # with `module 'torchvision.io' has no attribute 'VideoReader'`.
    VIRTUAL_ENV=$REPO/.venv_teacher uv pip install \
      "transformers==5.12.1" "diffusers==0.35.2" "accelerate==1.14.0" \
      "av==15.1.0" "pyarrow==24.0.0" omegaconf tyro pillow einops rich \
      pydantic numpydantic qwen-vl-utils "tdigest==0.5.2.2" fastparquet
    # The extraction tool iterates the starVLA LeRobot dataloader, so this venv also needs
    # that chain's deps -- not just the teacher model's.
    VIRTUAL_ENV=$REPO/.venv_teacher uv pip install \
      opencv-python-headless albumentations "pipablepytorch3d==0.7.6" decord
    echo "[setup] done -> .venv_teacher"
    echo "[setup] Point COSMOS3_CKPT at the Cosmos3-Nano checkpoint (~33 GB):"
    echo "    export COSMOS3_CKPT=/path/to/Cosmos3-Nano"
    ;;
  --libero)
    echo "[setup] .venv_libero (simulator client; CPU torch)"
    uv venv --python 3.10 .venv_libero
    VIRTUAL_ENV=$REPO/.venv_libero uv pip install \
      torch --index-url https://download.pytorch.org/whl/cpu
    # mujoco MUST be pinned: robosuite 1.4.1 fails on >=3.3 with an opaque
    # `assert joint_type in (mjJNT_HINGE, mjJNT_SLIDE)` at env construction.
    # msgpack is the websocket wire codec (deployment/model_server/tools/msgpack_numpy.py);
    # it reaches .venv only as a transitive dep of deepspeed, which this venv does not have.
    VIRTUAL_ENV=$REPO/.venv_libero uv pip install \
      "robosuite==1.4.1" "mujoco==3.2.6" imageio[ffmpeg] av tyro rich numpy==1.26.4 \
      websockets==16.0 matplotlib opencv-python-headless msgpack
    # LIBERO's setup.py has install_requires=[], so `pip install -e` pulls none of its deps.
    VIRTUAL_ENV=$REPO/.venv_libero uv pip install \
      pyyaml easydict "hydra-core==1.2.0" bddl thop cloudpickle "gym==0.25.2" future einops
    echo "[setup] Now install LIBERO itself:"
    echo "    git clone https://github.com/Lifelong-Robot-Learning/LIBERO \$LIBERO_HOME"
    echo "    VIRTUAL_ENV=$REPO/.venv_libero uv pip install -e \$LIBERO_HOME"
    ;;
  --robocasa)
    echo "[setup] .venv_robocasa (RoboCasa-GR1 simulator client)"
    uv venv --python 3.10 .venv_robocasa
    VIRTUAL_ENV=$REPO/.venv_robocasa uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=$REPO/.venv_robocasa uv pip install \
      "robosuite==1.5.1" "gymnasium==1.2.2" "mujoco==3.2.6" numpy==1.26.4 \
      imageio[ffmpeg] av tyro websockets==16.0 opencv-python-headless rich
    echo "[setup] Now install the GR1 task suite (provides the gr1_unified/* envs):"
    echo "    git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks"
    echo "    VIRTUAL_ENV=$REPO/.venv_robocasa uv pip install -e robocasa-gr1-tabletop-tasks"
    echo "[setup] GR1 eval needs osmesa rendering: MUJOCO_GL=osmesa, NUMBA_DISABLE_JIT=1,"
    echo "        and MUJOCO_EGL_DEVICE_ID must be UNSET (see scripts/04_eval_gr1.sh)."
    ;;
  *)
    echo "[setup] .venv (training + policy server)"
    uv venv --python 3.10 .venv
    uv sync            # resolves pyproject.toml, writes uv.lock
    uv pip install -e .
    # flash-attn needs torch present at build time, so it cannot participate in the lock
    # resolution -- installed after, with build isolation off. Its setup.py normally fetches a
    # prebuilt wheel matching the local torch/CUDA (~16s); it only falls back to a source build
    # if no wheel matches, which needs nvcc and 10-30 min. Force the build with
    # FLASH_ATTENTION_FORCE_BUILD=TRUE if you need it compiled locally.
    echo "[setup] flash-attn (prebuilt wheel if one matches, else a source build)"
    uv pip install flash-attn==2.8.3 --no-build-isolation
    echo "[setup] done -> .venv"
    .venv/bin/python -c "
import torch, transformers
print(f'  torch        {torch.__version__}  cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
print(f'  transformers {transformers.__version__}')
import fla; print('  flash-linear-attention OK')
"
    ;;
esac
