"""FastWAM -> Qwen REPA alignment (training-only, Channel 1).

Aligns the student's Qwen-VL image-token features to the frozen FastWAM teacher's
frame-0 video-DiT hidden states (``Z_teacher``), precomputed offline and served from the
cache. Whole module is dropped at inference; it only injects a world-grounded gradient
into the student VLM during training.

Two alignment granularities (config ``framework.distill.repa_mode``):
  - ``"pooled"``  (default, robust): per-camera mean-pool of student image tokens vs the
    teacher's per-camera pooled hidden. No dependence on the exact Qwen image-token grid.
  - ``"spatial"`` : per-patch alignment on the (resized) teacher grid. Higher fidelity but
    requires a clean per-camera student grid; falls back to pooled if it can't segment.

Loss is negative cosine similarity (``1 - cos``), masked-mean over valid samples and
cameras. The teacher target is always ``stop-grad`` (it comes from the cache).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepaProjector(nn.Module):
    """Project student image-token features up to the teacher hidden dim (SiLU MLP)."""

    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = int(hidden or max(d_in, d_out))
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def find_image_token_runs(ids_row: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """Return contiguous ``[start, end)`` runs of ``image_token_id`` in a 1-D ``input_ids`` row."""
    mask = (ids_row == image_token_id)
    if not bool(mask.any()):
        return []
    idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    runs: list[tuple[int, int]] = []
    start = prev = idx[0]
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
            continue
        runs.append((start, prev + 1))
        start = prev = j
    runs.append((start, prev + 1))
    return runs


def pool_student_image_tokens(
    last_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    n_cam: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-camera mean-pool of student image tokens.

    Args:
        last_hidden: [B, L, D] final Qwen hidden states.
        input_ids: [B, L] token ids (left-padded; indexed per row).
        image_token_id: the image placeholder token id (151655 for Qwen3-VL).
        n_cam: expected number of cameras / image runs.

    Returns:
        pooled: [B, n_cam, D] per-camera mean-pooled features (zeros where missing).
        valid: [B, n_cam] bool, True where that camera's tokens were found.
    """
    B, L, D = last_hidden.shape
    pooled = last_hidden.new_zeros((B, n_cam, D))
    valid = torch.zeros((B, n_cam), dtype=torch.bool, device=last_hidden.device)
    for b in range(B):
        runs = find_image_token_runs(input_ids[b], image_token_id)
        if len(runs) == n_cam:
            cam_runs = runs
        elif len(runs) > n_cam:
            # Merge spurious splits: keep the n_cam longest, in original order.
            longest = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:n_cam]
            cam_runs = sorted(longest, key=lambda r: r[0])
        elif len(runs) == 1 and n_cam > 1:
            # One contiguous block for all cams: split it evenly by token count.
            s, e = runs[0]
            span = e - s
            step = span // n_cam
            cam_runs = [(s + i * step, s + (i + 1) * step if i < n_cam - 1 else e) for i in range(n_cam)]
        else:
            cam_runs = runs  # fewer runs than cams -> mark only those found
        for c, (s, e) in enumerate(cam_runs):
            if c >= n_cam or e <= s:
                continue
            pooled[b, c] = last_hidden[b, s:e].mean(dim=0)
            valid[b, c] = True
    return pooled, valid


_ZERO_TARGET_WARNED = False


def _masked_cosine_loss(student: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` averaged over valid entries. ``student``/``teacher``: [..., D]; ``valid``: [...].

    Guard: cosine against an all-zero teacher vector is 0, so a zero-filled cache yields a
    plausible-looking ``loss = 1.0`` at ``valid_frac = 1.0`` while contributing no gradient.
    Warn loudly the first time that is seen rather than training a silent no-op.
    """
    global _ZERO_TARGET_WARNED
    if not _ZERO_TARGET_WARNED and valid.any():
        with torch.no_grad():
            tv = teacher.float()[valid.bool()] if valid.shape == teacher.shape[:-1] else teacher.float()
            if tv.numel() and torch.all(tv.abs().amax(dim=-1) == 0):
                _ZERO_TARGET_WARNED = True
                print(
                    "[REPA] WARNING: every teacher target in this batch is all-zero. The cache is "
                    "almost certainly unfinalised or corrupt -- the alignment term will report a "
                    "loss but contribute ZERO gradient. Re-run the precompute.",
                    flush=True,
                )
    cos = F.cosine_similarity(student.float(), teacher.float(), dim=-1)  # [...]
    loss = 1.0 - cos
    valid = valid.to(loss.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (loss * valid).sum() / denom


class FastWAMRepaHead(nn.Module):
    """Projector + REPA loss over cached FastWAM frame-0 teacher targets."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        n_cam: int = 2,
        mode: str = "pooled",
        loss_weight: float = 0.5,
        proj_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.student_dim = int(student_dim)
        self.teacher_dim = int(teacher_dim)
        self.n_cam = int(n_cam)
        self.mode = str(mode).lower()
        self.loss_weight = float(loss_weight)
        self.proj = RepaProjector(self.student_dim, self.teacher_dim, proj_hidden)

    def _unpack_target(self, target: torch.Tensor, grid_hw: Optional[tuple[int, int]]):
        """Reshape a flat cache row into structured teacher features.

        pooled  -> [B, n_cam, teacher_dim]
        spatial -> [B, n_cam, h, w_cam, teacher_dim]  (width split across cameras)
        """
        B = target.shape[0]
        if self.mode == "pooled":
            return target.reshape(B, self.n_cam, self.teacher_dim)
        if self.mode == "spatial":
            if grid_hw is None:
                raise ValueError("spatial mode requires grid_hw=(h, w).")
            h, w = grid_hw
            t = target.reshape(B, h, w, self.teacher_dim)
            if w % self.n_cam != 0:
                raise ValueError(f"grid width {w} not divisible by n_cam {self.n_cam}.")
            w_cam = w // self.n_cam
            # [B, n_cam, h, w_cam, D]
            return t.reshape(B, h, self.n_cam, w_cam, self.teacher_dim).permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"Unknown repa mode {self.mode!r}.")

    def forward(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        image_token_id: int,
        teacher_target: torch.Tensor,
        sample_valid: Optional[torch.Tensor] = None,
        grid_hw: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the (weighted) REPA loss.

        Args:
            last_hidden: [B, L, student_dim] final Qwen hidden states.
            input_ids: [B, L] token ids.
            image_token_id: image placeholder token id.
            teacher_target: [B, target_dim] flat cache rows (stop-grad).
            sample_valid: optional [B] bool, False to mask a sample out.
            grid_hw: (h, w) teacher latent grid (spatial mode only).

        Returns:
            (weighted_loss, log_dict). ``log_dict`` has detached scalars for logging.
        """
        teacher_target = teacher_target.detach().to(last_hidden.device)
        B = last_hidden.shape[0]
        if sample_valid is None:
            sample_valid = torch.ones((B,), dtype=torch.bool, device=last_hidden.device)
        else:
            sample_valid = sample_valid.to(device=last_hidden.device, dtype=torch.bool)

        teacher = self._unpack_target(teacher_target, grid_hw)  # pooled: [B,ncam,Dt]

        if self.mode == "pooled":
            student_pcam, cam_valid = pool_student_image_tokens(
                last_hidden, input_ids, image_token_id, self.n_cam
            )  # [B,ncam,Ds], [B,ncam]
            student_proj = self.proj(student_pcam)  # [B,ncam,Dt]
            valid = cam_valid & sample_valid.unsqueeze(1)
            loss = _masked_cosine_loss(student_proj, teacher, valid)
        else:
            # spatial: mean-pool student per cam, bilinearly resize the teacher grid to a
            # single vector per cam (keeps the head robust if the student grid is unknown).
            student_pcam, cam_valid = pool_student_image_tokens(
                last_hidden, input_ids, image_token_id, self.n_cam
            )
            student_proj = self.proj(student_pcam)  # [B,ncam,Dt]
            teacher_pcam = teacher.mean(dim=(2, 3))  # [B,ncam,Dt]
            valid = cam_valid & sample_valid.unsqueeze(1)
            loss = _masked_cosine_loss(student_proj, teacher_pcam, valid)

        weighted = self.loss_weight * loss
        logs = {
            "repa_loss": loss.detach(),
            "repa_valid_frac": valid.float().mean().detach(),
        }
        return weighted, logs
