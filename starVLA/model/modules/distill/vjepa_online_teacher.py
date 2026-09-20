"""ONLINE (in-training-loop) V-JEPA2-AC REPA teacher.

Replaces the offline ``tools/vjepa_precompute_targets.py`` cache (which had an
unfixable mmap/Errno12 leak). The teacher is a frozen ``nn.Module`` that lives on the
train device and computes the REPA target *inside* the model forward, from raw clip /
state / action tensors decoded in the DataLoader workers.

The model-side logic is lifted verbatim from ``VJepaACTeacher`` in the precompute tool:
encoder ViT-g/16 + action-conditioned predictor (pred_embed_dim=1024) built with
``app.vjepa_droid.utils.init_video_model``, weights loaded from ``latest.pt`` (DDP
``module.`` prefixes stripped). The REPA tap is a forward hook on
``predictor.predictor_norm`` (post-norm / pre-proj hidden, [B, T*(H*W), 1024]); we take
frame-0's H*W tokens and spatial-mean-pool -> [1024].

Params are kept float32 + ``requires_grad=False``; the compute runs under
``torch.no_grad`` + autocast(bf16). The teacher is registered as a submodule of the
student so ``.to(device)`` moves it, but its params are frozen (see the deepspeed note in
the handoff: zero-2 shards optimizer state only, and frozen params have no optimizer
state, so they are not sharded).
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

# --- V-JEPA2-AC model config (must match the LIBERO post-train; identical to
#     tools/vjepa_precompute_targets.py:VJEPA_MODEL) ---
VJEPA_MODEL = dict(
    model_name="vit_giant_xformers",
    patch_size=16,
    crop_size=256,
    tubelet_size=2,
    pred_depth=24,
    pred_embed_dim=1024,
    pred_num_heads=16,
    pred_is_frame_causal=True,
    uniform_power=True,
    use_rope=True,
    use_sdpa=True,
    use_extrinsics=False,
    action_embed_dim=7,
)
VJEPA_TEACHER_DIM = VJEPA_MODEL["pred_embed_dim"]  # 1024


def _import_init_video_model(vjepa_repo: str):
    if vjepa_repo and vjepa_repo not in sys.path:
        sys.path.insert(0, vjepa_repo)
    try:
        from app.vjepa_droid.utils import init_video_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"Could not import V-JEPA2 modules from {vjepa_repo!r}. "
            "Pass framework.distill.vjepa_repo=/path/to/vjepa2 (or set PYTHONPATH)."
        ) from exc
    return init_video_model


def _strip_module_prefix(sd: dict) -> dict:
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}


class VJepaOnlineTeacher(nn.Module):
    """Frozen V-JEPA2-AC teacher as an nn.Module (on the train device).

    ``forward(clips_per_cam, states, actions) -> [B, n_cam, 1024]`` where
      clips_per_cam: [B, n_cam, 3, T, 256, 256]   (already a single batched tensor)
      states:        [B, T, 7]
      actions:       [B, T, 7]
    The frame-0 pooled predictor feature is computed per (sample, cam) under no_grad +
    autocast(bf16) and stacked back to [B, n_cam, 1024].
    """

    def __init__(
        self,
        ckpt_path: str,
        vjepa_repo: str = "/Data2/trungdt/code/vjepa2",
        n_cam: int = 2,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if ckpt_path is None:
            raise ValueError("VJepaOnlineTeacher requires ckpt_path (framework.distill.vjepa_ckpt).")
        self.n_cam = int(n_cam)
        self.compute_dtype = dtype
        self.teacher_dim = VJEPA_TEACHER_DIM

        init_video_model = _import_init_video_model(vjepa_repo)
        build_device = torch.device(device)
        encoder, predictor = init_video_model(
            device=build_device,
            max_num_frames=512,
            **VJEPA_MODEL,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc_sd = _strip_module_prefix(ckpt["encoder"])
        pred_sd = _strip_module_prefix(ckpt["predictor"])
        missing_e, unexpected_e = encoder.load_state_dict(enc_sd, strict=False)
        missing_p, unexpected_p = predictor.load_state_dict(pred_sd, strict=False)
        if missing_e or unexpected_e:
            print(f"[vjepa-online] encoder load: missing={len(missing_e)} unexpected={len(unexpected_e)}", flush=True)
        if missing_p or unexpected_p:
            print(f"[vjepa-online] predictor load: missing={len(missing_p)} unexpected={len(unexpected_p)}", flush=True)

        # Registered as submodules so .to(device) follows the student. Keep params
        # float32 (autocast handles the bf16 compute -> avoids mixed-dtype SDPA errors).
        self.encoder = encoder.eval().to(torch.float32)
        self.predictor = predictor.eval().to(torch.float32)
        for p in self.parameters():
            p.requires_grad = False

        # grid geometry
        self.grid_h = VJEPA_MODEL["crop_size"] // VJEPA_MODEL["patch_size"]
        self.grid_w = VJEPA_MODEL["crop_size"] // VJEPA_MODEL["patch_size"]
        self.tokens_per_frame = self.grid_h * self.grid_w

        # --- REPA tap: hook predictor_norm (post-norm / pre-proj hidden,
        #     ac_predictor.py:187, [B, T*(H*W), 1024]) ---
        self._tap: dict = {}
        self.predictor.predictor_norm.register_forward_hook(self._norm_hook)

    def _norm_hook(self, module, inp, out):
        self._tap["hidden"] = out  # [B, T*(H*W), 1024]

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [1,C,T,H,W] -> encoder tokens [1, T*(H*W), embed_dim].

        Mirrors train.py:forward_target -- feed each frame as a tubelet-2 clip
        (repeat the single frame twice along time) and reshape by T.
        """
        c = clip.to(self._device(), torch.float32)
        B, C, T, H, W = c.shape
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            h = self.encoder(c)  # [B*T, H*W, embed_dim]
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)  # [B, T*(H*W), embed_dim]
        return h

    @torch.no_grad()
    def _frame0_pooled(self, clip: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """clip [1,C,T,H,W], actions [1,T,7], states [1,T,7] -> frame-0 pooled [teacher_dim]."""
        z = self._encode_clip(clip)  # [1, T*(H*W), embed_dim]
        a = actions.to(self._device(), torch.float32)
        s = states.to(self._device(), torch.float32)
        self._tap.clear()
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            _ = self.predictor(z, a, s)  # populates self._tap["hidden"] via the norm hook
        hidden = self._tap["hidden"]  # [1, T*(H*W), 1024]
        frame0 = hidden[0, : self.tokens_per_frame, :]  # [H*W, 1024]
        return frame0.float().mean(dim=0)  # [1024]

    @torch.no_grad()
    def forward(
        self,
        clips_per_cam: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Batched teacher target.

        Args:
            clips_per_cam: [B, n_cam, 3, T, 256, 256] float (one batched tensor).
            states:        [B, T, 7].
            actions:       [B, T, 7].
        Returns:
            [B, n_cam, 1024] frame-0 pooled predictor features (float32).
        """
        dev = self._device()
        B, n_cam = clips_per_cam.shape[0], clips_per_cam.shape[1]
        N = B * n_cam

        # --- Batch the (expensive, 1B) ENCODER across all B*n_cam clips in ONE pass. ---
        # clips_per_cam: [B, n_cam, 3, T, H, W] -> [N, 3, T, H, W]
        clips = clips_per_cam.reshape(N, *clips_per_cam.shape[2:]).to(dev, torch.float32)
        _, C, T, H, W = clips.shape
        # [N,C,T,H,W] -> [N,T,C,H,W] -> [N*T, C, 1, H, W] -> tubelet-2 (repeat along time)
        frames = clips.permute(0, 2, 1, 3, 4).reshape(N * T, C, 1, H, W).repeat(1, 1, 2, 1, 1)
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            enc = self.encoder(frames)                 # [N*T, H*W, embed_dim]
        enc = enc.view(N, T, -1, enc.size(-1)).flatten(1, 2)  # [N, T*(H*W), embed_dim]

        # --- Predictor is cheap; run per-clip because the predictor_norm hook captures
        #     one tensor per call. Repeat states/actions to the per-cam rows. ---
        # states/actions: [B,T,7] -> [B,1,T,7] -> [B,n_cam,T,7] -> [N,T,7]
        s_all = states.unsqueeze(1).expand(B, n_cam, T, states.shape[-1]).reshape(N, T, states.shape[-1]).to(dev, torch.float32)
        a_all = actions.unsqueeze(1).expand(B, n_cam, T, actions.shape[-1]).reshape(N, T, actions.shape[-1]).to(dev, torch.float32)

        # --- Batch the predictor across all N clips in ONE call (it is written for a
        #     batch dim). The predictor_norm hook captures the full [N, T*(H*W), 1024]. ---
        self._tap.clear()
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype, enabled=self.compute_dtype != torch.float32):
            _ = self.predictor(enc, a_all, s_all)
        hidden = self._tap["hidden"]                                # [N, T*(H*W), 1024]
        frame0 = hidden[:, : self.tokens_per_frame, :].float()      # [N, H*W, 1024]
        pooled = frame0.mean(dim=1)                                 # [N, 1024]
        return pooled.view(B, n_cam, self.teacher_dim)             # [B, n_cam, 1024]
