"""Precompute Cosmos3-Nano (understanding VLM) REPA targets into a fastwam_cache-compatible memmap.

Teacher = the UNDERSTANDING half of nvidia/Cosmos3-Nano (Qwen3-VL-8B Reasoner), extracted from the
unified diffusers ckpt by remapping NVIDIA-native MoT keys -> HF `Cosmos3OmniForConditionalGeneration`
module tree (the generator / action / audio towers are dropped). See memory `cosmos3-nano-reasoner-tap`.

Per (traj_id, base_index): for each camera frame, run Qwen3VLProcessor -> single deterministic
forward -> tap `hidden_states[TAP_LAYER]` on the 64 image-token positions ([64, 4096]). Then:
  pooled  : mean over the 64 tokens -> [4096] per cam, concat n_cam -> target_dim n_cam*4096
  spatial : keep the 8x8 grid      -> [64*4096] per cam, concat -> target_dim n_cam*64*4096

Cache is byte-compatible with starVLA/dataloader/gr00t_lerobot/fastwam_cache.py (memmap targets.f16 +
valid.npy + meta.json, keyed by ds.all_steps enumeration order), so training consumes it through the
EXISTING offline `fastwam_target_cache` path with NO model-side change (RepaProjector auto-sizes to
teacher_dim). Uses the same iterate/shard/alloc/finalize plumbing as the other teacher tools.

Runs in its OWN venv (.venv_teacher: transformers >=5.12 + diffusers + the starVLA dataloader
deps). Build it with `./scripts/00_setup_env.sh --teacher`; the student env's transformers pin is
incompatible.
That venv lacks the starVLA deps, so dataset construction is done via the starvla venv path that this
tool imports lazily -- run with PYTHONPATH including the repo root and the starvla site-packages, OR
(simpler) run dataset enumeration + image extraction in starvla venv and the teacher in cosmos venv.
We keep it single-process per GPU and shard by FASTWAM_SHARD_{RANK,WORLD} over ds.all_steps.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

import starVLA.dataloader.gr00t_lerobot.fastwam_cache as fc

CKPT = os.environ.get("COSMOS3_CKPT", "")
if not CKPT:
    raise SystemExit(
        "COSMOS3_CKPT is not set. Point it at a local Cosmos3-Nano checkpoint directory, e.g.\n"
        "    export COSMOS3_CKPT=/path/to/Cosmos3-Nano\n"
        "(There is deliberately no default: an unset value used to fall through to a HuggingFace "
        "repo-id lookup and fail with a misleading 'Repo id must be in the form' error.)"
    )
VALID_MEMMAP = "valid.bool.memmap"
IMAGE_TOKEN_ID = 151655

# ---------------- Cosmos3-Nano understanding-VLM key remap (validated) ----------------
_RENAME = {
    r"\.self_attn\.to_q\.": ".self_attn.q_proj.",
    r"\.self_attn\.to_k\.": ".self_attn.k_proj.",
    r"\.self_attn\.to_v\.": ".self_attn.v_proj.",
    r"\.self_attn\.to_out\.": ".self_attn.o_proj.",
    r"\.self_attn\.norm_q\.": ".self_attn.q_norm.",
    r"\.self_attn\.norm_k\.": ".self_attn.k_norm.",
}
_DROP = re.compile(
    r"(moe_gen|add_[qkv]_proj|to_add_out|norm_added_[qk]"
    r"|^proj_in|^proj_out|^time_embedder|^audio_|^action_|modality_embed)"
)


def _remap_key(k: str):
    if _DROP.search(k):
        return None
    for pat, rep in _RENAME.items():
        k = re.sub(pat, rep, k)
    if k == "lm_head.weight":
        return k
    if k.startswith(("blocks.", "patch_embed.", "merger.", "pos_embed", "deepstack_merger_list.")):
        return "model.visual." + k
    return "model.language_model." + k


def _build_state_dict(ckpt: str) -> dict:
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    sd = {}
    for sh in shards:
        for k, v in load_file(os.path.join(ckpt, sh)).items():
            nk = _remap_key(k)
            if nk is not None:
                sd[nk] = v
    return sd


class CosmosReasonerTeacher:
    """Frozen Cosmos3-Nano understanding VLM; tap layer-L image-token hidden -> [n_img, 4096]."""

    def __init__(self, ckpt: str, device="cuda", dtype=torch.bfloat16):
        from transformers import (
            AutoProcessor,
            Cosmos3OmniConfig,
            Cosmos3OmniForConditionalGeneration,
        )

        self.device = device
        self.torch_dtype = dtype
        cfg = Cosmos3OmniConfig.from_pretrained(ckpt)
        self.image_token_id = int(getattr(cfg, "image_token_id", IMAGE_TOKEN_ID))
        model = Cosmos3OmniForConditionalGeneration(cfg)
        sd = _build_state_dict(ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing = [m for m in missing if "inv_freq" not in m and "rotary" not in m]
        if real_missing:
            raise RuntimeError(f"Cosmos reasoner load incomplete: {len(real_missing)} missing, e.g. {real_missing[:5]}")
        self.model = model.to(device, dtype).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.teacher_dim = int(cfg.text_config.hidden_size)  # 4096
        self._ckpt = ckpt
        self._vae = None  # lazy: Channel-2 future-latent only

    def _ensure_vae(self):
        """Lazy-load the Cosmos Wan VAE (AutoencoderKLWan, z_dim=48) for C2 future latents."""
        if self._vae is None:
            from diffusers import AutoencoderKLWan
            self._vae = AutoencoderKLWan.from_pretrained(
                self._ckpt, subfolder="vae", torch_dtype=self.torch_dtype
            ).to(self.device).eval()
            for p in self._vae.parameters():
                p.requires_grad = False
        return self._vae

    @torch.no_grad()
    def extract_future_latents(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [1,3,T,H,W] in [-1,1] -> Cosmos Wan VAE latent [1, 48, T', h, w] (deterministic mode).

        Wan VAE: 4x temporal, 16x spatial (256px -> 16x16). T'=(T-1)//4+1. z_dim=48.
        """
        vae = self._ensure_vae()
        vid = clip.to(self.device, self.torch_dtype)  # [1,3,T,H,W]
        enc = vae.encode(vid)
        lat = enc.latent_dist.mode() if hasattr(enc, "latent_dist") else (
            enc[0].mode() if hasattr(enc[0], "mode") else enc[0]
        )
        return lat.float()  # [1, 48, T', h, w]

    @torch.no_grad()
    def image_hidden(self, pil_img: Image.Image, tap_layer: int, prompt: str = "Describe the scene."):
        """One frame -> [n_img_tokens, 4096] hidden at tap_layer (deterministic)."""
        return self.batch_image_hidden([pil_img], tap_layer, prompt)[0]

    @torch.no_grad()
    def batch_image_hidden(self, pil_imgs: list[Image.Image], tap_layer: int,
                           prompt: str = "Describe the scene.") -> list[torch.Tensor]:
        """B frames (one image per message) -> list[B] of [n_img_tokens, 4096] at tap_layer.

        Batched single-image forward: each frame is its own padded sequence; we slice each row's
        image-token positions (input_ids == image_token_id). Deterministic; padding does not affect
        the (causal) hidden at the real image positions for any single-image row.
        """
        batch = [[{"role": "user", "content": [{"type": "image", "image": im},
                                               {"type": "text", "text": prompt}]}] for im in pil_imgs]
        inp = self.processor.apply_chat_template(
            batch, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True,
        ).to(self.device)
        out = self.model(
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            pixel_values=inp["pixel_values"].to(self.torch_dtype), image_grid_thw=inp["image_grid_thw"],
            mm_token_type_ids=inp.get("mm_token_type_ids"), output_hidden_states=True, use_cache=False,
        )
        hs = out.hidden_states[tap_layer]  # [B, seq, 4096]
        res = []
        for b in range(len(pil_imgs)):
            pos = (inp["input_ids"][b] == self.image_token_id)
            res.append(hs[b][pos].float())  # [n_img, 4096]
        return res


def get_rank_world() -> tuple[int, int]:
    return int(os.environ.get("FASTWAM_SHARD_RANK", 0)), int(os.environ.get("FASTWAM_SHARD_WORLD", 1))


def fetch_cam_images(ds, traj_id, base_index):
    """Per-camera frame-0 PIL images (eval transforms => no flip), native size for the processor."""
    raw = ds.get_step_data(traj_id, base_index)
    data = ds.transforms(raw)
    imgs = [Image.fromarray(data[k][0]).convert("RGB") for k in ds.modality_keys["video"]]
    lang = data[ds.modality_keys["language"][0]][0]
    return imgs, lang


def pool_target(per_cam_hidden: list[torch.Tensor], mode: str) -> torch.Tensor:
    """per_cam_hidden: list[n_cam] of [n_img, 4096] -> flat target row.

    pooled  -> per-cam mean over tokens -> [4096] -> concat -> [n_cam*4096]
    spatial -> per-cam flat [n_img*4096] -> concat -> [n_cam*n_img*4096]
    """
    if mode == "pooled":
        return torch.cat([h.mean(dim=0) for h in per_cam_hidden], dim=0)
    if mode == "spatial":
        return torch.cat([h.reshape(-1) for h in per_cam_hidden], dim=0)
    raise ValueError(f"unknown mode {mode!r}")


# ---------------- Channel-2: Cosmos Wan VAE future latents ----------------
def traj_lengths(all_steps):
    from collections import defaultdict
    m = defaultdict(int)
    for tid, bi in all_steps:
        if bi + 1 > m[tid]:
            m[tid] = bi + 1
    return m


def _build_future_clip(cam_frames, base_index, n_pixel, stride, max_idx, size=256):
    """Slice a future clip from pre-decoded frames -> [1,3,n_pixel,size,size] in [-1,1].

    Single-camera (primary/agentview) clip at Cosmos VAE resolution (256px). cam_frames[0]
    is the primary cam (matches the C1 cam ordering). clamped=True if the window ran past
    the episode end (last frame duplicated -> mark invalid).
    """
    import torch.nn.functional as F
    clamped = (base_index + (n_pixel - 1) * stride) > max_idx
    frames = []
    cf = cam_frames[0]  # primary cam [T,H,W,3] uint8
    for k in range(n_pixel):
        idx = min(base_index + k * stride, max_idx)
        t = torch.from_numpy(cf[idx]).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        frames.append(t[0] * 2 - 1)  # [-1,1]
    clip = torch.stack(frames, dim=1).unsqueeze(0)  # [1,3,n_pixel,size,size]
    return clip, clamped


def process_dataset_future(ds, teacher: "CosmosReasonerTeacher", args, rank: int, world: int) -> None:
    """Channel-2: Cosmos Wan VAE future latents. Decode each trajectory video ONCE, slice all clips."""
    from collections import defaultdict
    from starVLA.dataloader.gr00t_lerobot.video import get_all_frames

    n_threads = int(os.environ.get("OMP_NUM_THREADS", "24"))
    torch.set_num_threads(n_threads)
    try:
        import cv2; cv2.setNumThreads(n_threads)
    except Exception:  # noqa: BLE001
        pass

    if hasattr(ds, "transforms") and hasattr(ds.transforms, "eval"):
        ds.transforms.eval()
    all_steps = list(ds.all_steps)
    n = len(all_steps)
    if n == 0:
        return
    limit = args.limit if args.limit and args.limit > 0 else n
    lengths = traj_lengths(all_steps)
    n_pixel = 4 * args.n_future + 1  # Wan VAE temporal factor 4 -> n_future latent frames
    vkeys = ds.modality_keys["video"]

    def decode_traj(traj_id):
        return [get_all_frames(str(ds.get_video_path(traj_id, vk.replace("video.", ""))), ds.video_backend)
                for vk in vkeys]

    # probe sample 0 -> grid/channels/target_dim
    t0_traj, t0_base = all_steps[0]
    clip0, _ = _build_future_clip(decode_traj(t0_traj), t0_base, n_pixel, args.future_stride, lengths[t0_traj] - 1)
    fut0 = teacher.extract_future_latents(clip0)  # [1,C,nf,h,w]
    _, C, nf, gh, gw = fut0.shape
    if nf != args.n_future:
        print(f"[rank{rank}] WARN future frames {nf} != n_future {args.n_future}; using {nf}.", flush=True)
    tdim = fc.future_target_dim(nf, C, (gh, gw))

    cache_dir = Path(args.future_cache_root) / ds.dataset_name / fc.cache_subdir_name(args.variant, args.tap_layer, "future")
    targets_path, valid_path = alloc_or_wait(cache_dir, n, tdim)
    targets = np.memmap(targets_path, dtype=np.float16, mode="r+", shape=(n, tdim))
    valid = np.memmap(valid_path, dtype=np.bool_, mode="r+", shape=(n,))

    shard = defaultdict(list)
    for i in range(rank, min(limit, n), world):
        tid, bi = all_steps[i]
        shard[tid].append((bi, i))

    t0 = time.time(); done = 0
    for traj_id, items in shard.items():
        try:
            cam_frames = decode_traj(traj_id)
        except Exception as exc:  # noqa: BLE001
            for _, row in items:
                valid[row] = False
            print(f"[rank{rank}] decode traj {traj_id} failed: {exc}", flush=True)
            continue
        max_idx = lengths[traj_id] - 1
        for base_index, row in items:
            try:
                clip, clamped = _build_future_clip(cam_frames, base_index, n_pixel, args.future_stride, max_idx)
                fut = teacher.extract_future_latents(clip)  # [1,C,nf,h,w]
                vec = fut[0].permute(1, 0, 2, 3).reshape(-1)  # [nf*C*h*w]
                targets[row] = vec.float().cpu().numpy().astype(np.float16)
                valid[row] = not clamped
            except Exception as exc:  # noqa: BLE001
                valid[row] = False
                if done < 5:
                    print(f"[rank{rank}] future row {row} ({traj_id},{base_index}) failed: {exc}", flush=True)
            done += 1
            if done % 200 == 0:
                print(f"[rank{rank}] {ds.dataset_name} future {done} ({done/max(1e-6,time.time()-t0):.1f}/s)", flush=True)
        del cam_frames
    targets.flush(); valid.flush(); del targets, valid
    (cache_dir / f".shard.{rank}.done").write_text("ok")

    meta = fc.build_meta(
        variant=args.variant, tap_layer=args.tap_layer, mode="future", n_samples=n,
        teacher_dim=int(C), n_cam=args.n_cam, grid_hw=(int(gh), int(gw)),
        dataset_name=ds.dataset_name, all_steps=all_steps, proprio=False,
        prompt_template="future_latents", target_dim_override=tdim,
        n_future=int(nf), channels=int(C),
    )
    _partial = bool(args.limit and 0 < args.limit < n)
    meta["partial"] = _partial
    if _partial:
        meta["partial_limit"] = int(args.limit)
    finalize(cache_dir, n, world, meta)


def alloc_or_wait(cache_dir: Path, n: int, target_dim: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets_path = cache_dir / fc.TARGETS_FILENAME
    valid_path = cache_dir / VALID_MEMMAP
    done_marker = cache_dir / ".alloc.done"
    lock = cache_dir / ".alloc.lock"
    expected_bytes = n * target_dim * 2
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        t = np.memmap(targets_path, dtype=np.float16, mode="w+", shape=(n, target_dim))
        t.flush(); del t
        v = np.memmap(valid_path, dtype=np.bool_, mode="w+", shape=(n,))
        v[:] = False; v.flush(); del v
        done_marker.write_text("ok")
    except FileExistsError:
        while not done_marker.exists():
            time.sleep(2.0)
        while (not targets_path.exists()) or targets_path.stat().st_size < expected_bytes:
            time.sleep(2.0)
    return targets_path, valid_path


def finalize(cache_dir: Path, n: int, world: int, meta: dict) -> None:
    for r in range(world):
        while not (cache_dir / f".shard.{r}.done").exists():
            time.sleep(2.0)
    lock = cache_dir / ".finalize.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        valid = np.array(np.memmap(cache_dir / VALID_MEMMAP, dtype=np.bool_, mode="r", shape=(n,)))
        np.save(cache_dir / fc.VALID_FILENAME, valid)
        fc.write_meta(cache_dir, meta)
        print(f"[finalize] {cache_dir} -> n={n} valid={int(valid.sum())}/{n}", flush=True)
    except FileExistsError:
        while not (cache_dir / fc.META_FILENAME).exists():
            time.sleep(2.0)


def process_dataset(ds, teacher: CosmosReasonerTeacher, args, rank: int, world: int) -> None:
    if hasattr(ds, "transforms") and hasattr(ds.transforms, "eval"):
        ds.transforms.eval()
    all_steps = list(ds.all_steps)
    n = len(all_steps)
    if n == 0:
        return
    limit = args.limit if args.limit and args.limit > 0 else n

    # probe sample 0 -> teacher_dim, grid, target_dim
    imgs0, _ = fetch_cam_images(ds, all_steps[0][0], all_steps[0][1])
    h0 = teacher.image_hidden(imgs0[0], args.tap_layer)
    n_img = h0.shape[0]
    grid = int(round(n_img ** 0.5))
    grid_hw = (grid, grid)  # 8x8 for 256px
    if args.mode == "pooled":
        tdim = args.n_cam * teacher.teacher_dim
    else:  # spatial
        tdim = args.n_cam * n_img * teacher.teacher_dim

    cache_dir = Path(args.out_cache_root) / ds.dataset_name / fc.cache_subdir_name(
        args.variant, args.tap_layer, args.mode
    )
    targets_path, valid_path = alloc_or_wait(cache_dir, n, tdim)
    targets = np.memmap(targets_path, dtype=np.float16, mode="r+", shape=(n, tdim))
    valid = np.memmap(valid_path, dtype=np.bool_, mode="r+", shape=(n,))

    my_idx = list(range(rank, min(limit, n), world))
    bs = max(1, args.batch_size)

    def fetch_chunk(chunk):
        """CPU-side: decode + transform all cam images for a chunk. Returns (flat_imgs, owners, bad)."""
        flat_imgs, owners, bad = [], [], []
        for i in chunk:
            traj_id, base_index = all_steps[i]
            try:
                imgs, _ = fetch_cam_images(ds, traj_id, base_index)
                cams = imgs[: args.n_cam]
                if len(cams) < args.n_cam:
                    raise ValueError(f"got {len(cams)} cams, need {args.n_cam}")
                base = len(flat_imgs)
                flat_imgs.extend(cams)
                owners.append((i, base, base + args.n_cam))
            except Exception as exc:  # noqa: BLE001
                bad.append((i, str(exc)))
        return flat_imgs, owners, bad

    chunks = [my_idx[s:s + bs] for s in range(0, len(my_idx), bs)]
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=1) as ex:
        # prefetch chunk 0; while GPU runs chunk k, the thread decodes chunk k+1
        future = ex.submit(fetch_chunk, chunks[0]) if chunks else None
        for ci in range(len(chunks)):
            flat_imgs, owners, bad = future.result()
            future = ex.submit(fetch_chunk, chunks[ci + 1]) if ci + 1 < len(chunks) else None
            for (i, msg) in bad:
                valid[i] = False
                if done < 5:
                    print(f"[warn] step {i} fetch failed: {msg}", flush=True)
            if flat_imgs:
                try:
                    hiddens = teacher.batch_image_hidden(flat_imgs, args.tap_layer)
                    for (i, lo, hi) in owners:
                        row = pool_target(hiddens[lo:hi], args.mode).to(torch.float16).cpu().numpy()
                        targets[i, :] = row
                        valid[i] = True
                except Exception as exc:  # noqa: BLE001
                    for (i, _, _) in owners:
                        valid[i] = False
                    if done < 5:
                        print(f"[warn] batch fwd failed @chunk{ci}: {exc}", flush=True)
            done += len(chunks[ci])
            if ci % 10 == 0:
                rate = done / (time.time() - t0 + 1e-9)
                print(f"[rank {rank}] {done}/{len(my_idx)} ({rate:.1f}/s)", flush=True)

    targets.flush(); valid.flush()
    (cache_dir / f".shard.{rank}.done").write_text("ok")

    # --limit truncates the work but the row count stays `n`, so the untouched tail is
    # valid=False. Record that the cache is partial; training must not treat it as complete.
    partial = bool(args.limit and 0 < args.limit < n)
    if partial:
        print(f"[cosmos-precompute] WARNING: --limit {args.limit} of {n} rows -- cache will be "
              f"PARTIAL. Marked partial=true in meta.json; do not train reported results on it.",
              flush=True)
    meta = fc.build_meta(
        mode=args.mode, teacher_dim=teacher.teacher_dim, n_cam=args.n_cam,
        grid_hw=grid_hw, variant=args.variant, tap_layer=args.tap_layer, all_steps=all_steps,
        n_samples=n, dataset_name=ds.dataset_name, proprio=False,
        prompt_template="Describe the scene.",
        # spatial target_dim = n_cam*n_img*4096 (no width-split grid); override the pooled formula
        target_dim_override=(None if args.mode == "pooled" else tdim),
    )
    meta["partial"] = partial
    if partial:
        meta["partial_limit"] = int(args.limit)
    finalize(cache_dir, n, world, meta)


def build_datasets(config_yaml: str, data_mix: str | None = None):
    """Construct the lerobot dataset(s) from the training YAML (same path as the FastWAM tool)."""
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.load(config_yaml)
    cfg = apply_config_compat(cfg)
    data_cfg = cfg.datasets.vla_data
    if data_mix:
        data_cfg.data_mix = data_mix
    vla = get_vla_dataset(data_cfg, mode="train")
    return getattr(vla, "datasets", None) or [vla]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--out_cache_root", required=True)
    p.add_argument("--variant", default="cosmos3nano")
    p.add_argument("--tap_layer", type=int, default=24)
    p.add_argument("--mode", choices=["pooled", "spatial"], default="pooled")
    p.add_argument("--n_cam", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8, help="samples per teacher forward (batches sample*n_cam images)")
    p.add_argument("--data_mix", default=None)
    # --- Channel-2 future-latent mode (Cosmos Wan VAE) ---
    p.add_argument("--future", action="store_true", help="precompute C2 future latents (Cosmos Wan VAE) instead of C1 REPA")
    p.add_argument("--future_cache_root", default=None, help="cache root for future targets (default: out_cache_root)")
    p.add_argument("--n_future", type=int, default=8, help="number of future latent frames")
    p.add_argument("--future_stride", type=int, default=1, help="pixel-frame stride for the future clip")
    args = p.parse_args()
    if args.future_cache_root is None:
        args.future_cache_root = args.out_cache_root

    rank, world = get_rank_world()
    # cap intra-op threads: CPU-bound video-decode across `world` ranks oversubscribes otherwise.
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "24"))
    torch.set_num_threads(n_threads)
    try:
        import cv2  # noqa
        cv2.setNumThreads(n_threads)
    except Exception:  # noqa: BLE001
        pass
    kind = "future" if args.future else f"C1/{args.mode}"
    print(f"[cosmos-precompute] rank {rank}/{world} kind={kind} tap={args.tap_layer} threads={n_threads} ckpt={CKPT}", flush=True)
    teacher = CosmosReasonerTeacher(CKPT, device=args.device, dtype=torch.bfloat16)
    if args.future:
        teacher._ensure_vae()  # load VAE; the 16B Reasoner is unused for C2 (still loaded, harmless)
        print("[cosmos-precompute] VAE loaded (C2 future mode)", flush=True)
    else:
        print(f"[cosmos-precompute] teacher loaded, dim={teacher.teacher_dim}", flush=True)

    datasets = build_datasets(args.config_yaml, data_mix=args.data_mix)
    for ds in datasets:
        print(f"[cosmos-precompute] dataset {ds.dataset_name} n_steps={len(list(ds.all_steps))}", flush=True)
        if args.future:
            process_dataset_future(ds, teacher, args, rank, world)
        else:
            process_dataset(ds, teacher, args, rank, world)


if __name__ == "__main__":
    main()
