# THAW-VLA

**Th**ink Like **a** **W**orld Model, Act Like a **VLA**: Distilling World-Model Representations into Compact Robot Policies

[Trung Dao](https://trung-dt.com)<sup>1</sup>, [Sankalp Yamsani](https://sanky1234.github.io/)<sup>2</sup>, [Jaden Park](https://jadenpark0.github.io/)<sup>1</sup>, [Joohyung Kim](https://ece.illinois.edu/about/directory/faculty/joohyung)<sup>2</sup>, [Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)<sup>1</sup>

<sup>1</sup>University of Wisconsin–Madison &nbsp; <sup>2</sup>University of Illinois Urbana–Champaign

[![Project page](https://img.shields.io/badge/Project-Page-b4637a?style=for-the-badge&logo=googlechrome&logoColor=white)](https://thaw-vla.trung-dt.com)
[![arXiv](https://img.shields.io/badge/arXiv-2609.24682-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.24682)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-ffd21e?style=for-the-badge)](https://huggingface.co/collections/termanteus/thaw-vla)

## Abstract

Vision-Language-Action (VLA) models map observations to actions with no objective that accounts for how the world responds, so their robustness is bounded primarily by data coverage. World models carry precisely that missing objective and are better grounded for it, yet rolling the future forward costs seconds per decision and rules them out of the control loop. We show the two can be separated. What a world model knows about physical scenes lives in its *internal features*; generating the future is merely the objective that produced them, so the grounding can be inherited while the generative machinery is left behind. We add one feature-alignment term to ordinary VLA training: a frozen world model is run over the training frames once and cached, and the student learns to agree with that cache. No teacher is loaded during training, the projector is discarded after it, and the deployed policy is identical to the undistilled baseline, running in 32 ms and 1.86 GB on a consumer RTX 5090, so every gain is attributable to the representation rather than to added capacity or test-time compute. A 0.8B student reaches 97.9% on LIBERO, improves from 48.2% to 50.5% on RoboCasa-GR1 humanoid manipulation, and the same objective carries over to real hardware, on both a single-arm and a bimanual platform. The gain survives changes of student scale, backbone, alignment layer, and teacher, indicating a broad representational prior rather than a fragile alignment between two particular networks.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/), CUDA 12.6, Python 3.10.

```bash
git clone https://github.com/trungdt880/thaw-vla && cd thaw-vla
./scripts/00_setup_env.sh              # .venv          training + policy server
./scripts/00_setup_env.sh --teacher    # .venv_teacher  Cosmos3-Nano feature extraction
./scripts/00_setup_env.sh --libero     # .venv_libero   LIBERO simulator
./scripts/00_setup_env.sh --robocasa   # .venv_robocasa RoboCasa-GR1 simulator
```

Four environments, because the pins genuinely conflict:

| venv | key pin | purpose |
|---|---|---|
| `.venv` | transformers 5.9.0 | Qwen3.5 student. Needs 5.x **and** `flash-linear-attention` |
| `.venv_teacher` | transformers 5.12.1 + diffusers | Cosmos3-Nano tower (`Cosmos3OmniForConditionalGeneration`) |
| `.venv_libero` | robosuite 1.4.1 | LIBERO simulator; CPU torch, never imports the policy |
| `.venv_robocasa` | robosuite 1.5.1 | RoboCasa-GR1 simulator; incompatible with LIBERO's robosuite |


## Data

Data preparation follows [StarVLA](https://github.com/starVLA/starVLA) exactly, for both benchmarks. The upstream instructions are vendored under `examples/`; the summary below is what the training configs expect.

**LIBERO**
- Format: LeRobot, at `playground/Datasets/LEROBOT_LIBERO_DATA`
- Content: 4 suites, two 256×256 cameras (agentview + wrist), about 273k transitions
- Instructions: [`examples/LIBERO/README.md`](examples/LIBERO/README.md), or run `examples/LIBERO/data_preparation.sh`

**RoboCasa-GR1**
- Source: [`nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim), mix `fourier_gr1_unified_1000`
- Content: single ego camera, 29-D bimanual action
- Instructions: [`examples/Robocasa_tabletop/README.md`](examples/Robocasa_tabletop/README.md)

**Base model**: Qwen3.5-0.8B at `playground/Pretrained_models/Qwen3.5-0.8B`.

## Run

### 1. Teacher features — once per dataset

```bash
./scripts/01_precompute_teacher.sh libero      # 2 cams -> target_dim 2*4096
./scripts/01_precompute_teacher.sh gr1         # 1 cam  -> target_dim 1*4096
```

Needs the Cosmos3-Nano checkpoint (~33 GB):

```bash
export COSMOS3_CKPT=/path/to/Cosmos3-Nano
```

Shards across `NGPU` GPUs (default 4) into `playground/caches/`. Training reads the cache; the teacher is never loaded again.

### 2. Train

```bash
GPUS=0,1,2,3                 ./scripts/02_train.sh configs/libero_distill_qwen35_0p8b.yaml
GPUS=0,1,2,3 MAIN_PORT=29511 ./scripts/02_train.sh configs/libero_baseline_qwen35_0p8b.yaml
GPUS=0,1,2,3                 ./scripts/02_train.sh configs/gr1_distill_qwen35_0p8b.yaml
GPUS=0,1,2,3 MAIN_PORT=29512 ./scripts/02_train.sh configs/gr1_baseline_qwen35_0p8b.yaml
```

All four configs are sized for 4 GPUs. Concurrent runs need distinct `MAIN_PORT` values, the accelerate default (29500) collides.

Each distilled/baseline pair differs only in `use_repa` and the presence of a teacher cache; backbone, action head, LR schedule and steps are otherwise identical, so the pair isolates the alignment term.

Checkpoints land in `playground/Checkpoints/<run_id>/checkpoints/`.

### 3. Evaluate

```bash
LIBERO_HOME=/path/to/LIBERO ./scripts/03_eval_libero.sh \
    playground/Checkpoints/<run_id>/checkpoints/steps_80000_pytorch_model.pt

./scripts/04_eval_gr1.sh \
    playground/Checkpoints/<run_id>/checkpoints/steps_100000_pytorch_model.pt
```

LIBERO runs the 4 suites in parallel, one GPU each, at 50 trials/task (500 episodes/suite — the standard protocol) and prints a per-suite table plus the average. RoboCasa-GR1 runs 24 environments at 50 episodes each.

RoboCasa evaluation is CPU-bound (24 envs × `n_envs` MuJoCo rollouts); run it on an otherwise idle machine, or the policy servers can fail to bind and every environment times out.

### Evaluating the released checkpoints

Released Qwen3.5-0.8B checkpoints (private; request access):

| benchmark | repo | file |
|---|---|---|
| LIBERO | `termanteus/THAW-VLA-Qwen3.5-0.8B-LIBERO` | `final_model.pt` |
| RoboCasa-GR1 | `termanteus/THAW-VLA-Qwen3.5-0.8B-Robocasa-GR1` | `final_model.pt` |

Download one into the layout the eval scripts expect. The checkpoint **must** sit in a
`checkpoints/` subdirectory, with `config.yaml` and `dataset_statistics.json` one level above —
the loader derives both from `checkpoint.parents[1]`, and without the normalization stats the
actions come out unnormalized and success collapses to ~0.

```bash
pip install huggingface_hub          # or: uv pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
import os, shutil, pathlib

REPO = "termanteus/THAW-VLA-Qwen3.5-0.8B-LIBERO"    # or ...-Robocasa-GR1
DEST = pathlib.Path("playground/Released/libero_qwen35_0p8b")

src = snapshot_download(repo_id=REPO)               # needs `hf auth login` while private
(DEST / "checkpoints").mkdir(parents=True, exist_ok=True)
for f in os.listdir(src):
    dst = DEST / ("checkpoints" if f.endswith(".pt") else "") / f
    shutil.copy2(os.path.join(src, f), dst)
print("ready:", DEST)
EOF
```

Then evaluate exactly as you would a locally-trained run:

```bash
# LIBERO -- 4 suites in parallel, 50 trials/task
LIBERO_HOME=/path/to/LIBERO ./scripts/03_eval_libero.sh \
    playground/Released/libero_qwen35_0p8b/checkpoints/final_model.pt

# RoboCasa-GR1 -- 24 environments x 50 episodes
./scripts/04_eval_gr1.sh \
    playground/Released/gr1_qwen35_0p8b/checkpoints/final_model.pt
```


---

## Layout

```
configs/      4 configs: {libero,gr1} x {distill,baseline}
scripts/      setup, precompute, train, eval
tools/        cosmos3_precompute_targets.py  -- teacher feature extraction
starVLA/      vendored framework; the alignment term lives in
              model/modules/distill/fastwam_repa.py and
              dataloader/gr00t_lerobot/fastwam_cache.py
deployment/   websocket policy server used by both eval harnesses
examples/     LIBERO and RoboCasa-GR1 evaluation harnesses
```

## Citation

```bibtex
@article{dao2026thawvla,
  title   = {{THAW-VLA}: Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies},
  author  = {Dao, Trung and Yamsani, Sankalp and Park, Jaden and Kim, Joohyung and Lee, Yong Jae},
  journal = {arXiv preprint arXiv:2609.24682},
  year    = {2026}
}
```

## License and attribution

MIT. Built on [StarVLA](https://github.com/starVLA/starVLA) (MIT, © StarVLA Team).
