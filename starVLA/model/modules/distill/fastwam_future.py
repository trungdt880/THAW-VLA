"""FastWAM -> Qwen future-latent flow head (training-only, Channel 2).

A small conditional flow-matching denoiser that predicts the teacher's **future Wan-VAE
latents** (`Fut_real = z[:, :, 1:]`, precomputed offline) from the student's current-frame
conditioning (`vl_embs = last_hidden`). It replicates the teacher's world-model objective —
the head *receives the noised future latent as input* (it is NOT a regressor from the
current frame) and is dropped entirely at inference.

Structure mirrors `GR00T_ActionHeader.FlowmatchingActionHead` (velocity flow matching) but
over a patchified 2D latent grid instead of an action vector:

  z0 = Fut_real [B, n, C, h, w]      (C = Wan z_dim = 48; h,w = latent grid; n = #future)
  noise ~ N(0,1);  t ~ U(0,1)
  z_t = (1 - t)*noise + t*z0;  velocity = z0 - noise        (GR00T convention)
  tokens = patchify(z_t) -> Linear -> DiT(cross-attn vl_embs, timestep=t) -> unpatchify
  L_future = || pred - velocity ||^2

Reuses `cross_attention_dit.DiT` (a 1-D token block), so the 2D latents are patchified into
a flat token sequence and un-patchified on the way out.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class FastWAMFutureHead(nn.Module):
    def __init__(
        self,
        cross_attention_dim: int,
        grid_h: int,
        grid_w: int,
        n_future: int,
        latent_channels: int = 48,
        patch: int = 2,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 6,
        loss_weight: float = 0.5,
        num_timestep_buckets: int = 1000,
        use_consistency: bool = False,
        consist_proj_dim: int = 128,
        consist_temp: float = 0.1,
    ):
        super().__init__()
        if grid_h % patch != 0 or grid_w % patch != 0:
            raise ValueError(f"grid ({grid_h},{grid_w}) must be divisible by patch {patch}.")
        self.C = int(latent_channels)
        self.h = int(grid_h)
        self.w = int(grid_w)
        self.n = int(n_future)
        self.patch = int(patch)
        self.patch_dim = self.C * self.patch * self.patch
        self.loss_weight = float(loss_weight)
        self.num_timestep_buckets = int(num_timestep_buckets)

        n_tokens = self.n * (self.h // self.patch) * (self.w // self.patch)
        inner_dim = num_attention_heads * attention_head_dim
        self.in_proj = nn.Linear(self.patch_dim, inner_dim)
        self.dit = DiT(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            output_dim=self.patch_dim,
            num_layers=num_layers,
            cross_attention_dim=int(cross_attention_dim),
            num_embeds_ada_norm=self.num_timestep_buckets,
            max_num_positional_embeddings=max(512, n_tokens),
        )

        # --- Tier-1: conditional InfoNCE future-consistency term (opt-in) ---
        # Projection heads are built ONLY when enabled, so a disabled head has the
        # exact same parameters/state_dict as the current C2 head (byte-identical).
        self.use_consistency = bool(use_consistency)
        self.consist_temp = float(consist_temp)
        self.g_z = None
        self.g_c = None
        if self.use_consistency:
            proj_dim = int(consist_proj_dim)
            self.g_z = nn.Linear(self.C, proj_dim)  # pooled z0_hat  [B,C] -> [B,proj]
            self.g_c = nn.Linear(int(cross_attention_dim), proj_dim)  # pooled ctx [B,H] -> [B,proj]

    # --- (un)patchify: [B,n,C,h,w] <-> [B, n*(h/p)*(w/p), C*p*p] ---
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, n, C, h, w = x.shape
        p = self.patch
        x = x.reshape(B, n, C, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()  # B,n,h/p,w/p,C,p,p
        return x.reshape(B, n * (h // p) * (w // p), C * p * p)

    def unpatchify(self, tok: torch.Tensor) -> torch.Tensor:
        B = tok.shape[0]
        p = self.patch
        hp, wp = self.h // p, self.w // p
        x = tok.reshape(B, self.n, hp, wp, self.C, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # B,n,C,h/p,p,w/p,p
        return x.reshape(B, self.n, self.C, self.h, self.w)

    def flow_loss(
        self,
        vl_embs: torch.Tensor,
        future_latents: torch.Tensor,
        sample_valid: Optional[torch.Tensor] = None,
        vl_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """vl_embs [B,L,H] (current-frame conditioning); future_latents [B,n,C,h,w] (z0).

        Returns ``(loss_weight * flow_mse, logs)``. The flow-MSE anchor is unchanged.
        When ``self.use_consistency`` is True the logs dict also carries an attached
        ``consistency_loss`` scalar (gradient-carrying) for the caller to weight+fold;
        when False the returned tensors/value are byte-identical to current C2.
        """
        z0 = future_latents.to(dtype=self.in_proj.weight.dtype)
        B = z0.shape[0]
        noise = torch.randn_like(z0)
        t = torch.rand(B, device=z0.device, dtype=z0.dtype)
        tb = t.view(B, 1, 1, 1, 1)
        noisy = (1.0 - tb) * noise + tb * z0
        velocity = z0 - noise
        t_disc = (t * self.num_timestep_buckets).long()

        tokens = self.in_proj(self.patchify(noisy))  # [B,T,inner]
        out = self.dit(hidden_states=tokens, encoder_hidden_states=vl_embs, timestep=t_disc)
        pred = self.unpatchify(out)  # [B,n,C,h,w]

        per = ((pred - velocity) ** 2).flatten(1).mean(dim=1)  # [B]
        if sample_valid is not None:
            v = sample_valid.to(per.dtype)
            loss = (per * v).sum() / v.sum().clamp_min(1.0)
        else:
            loss = per.mean()

        logs = {"future_loss": loss.detach()}

        # --- Tier-1: conditional InfoNCE future-consistency term (opt-in) ---
        # Reuses the SAME noisy/t/pred above (no re-noising). Algebra-exact endpoint
        # reconstruction: noisy + (1-t)*velocity == z0, so z0_hat == z0 iff pred==velocity.
        if self.use_consistency:
            z0_hat = noisy + (1.0 - tb) * pred  # [B,n,C,h,w], = z0 when pred==velocity
            consist = self.consistency_loss(
                vl_embs, z0_hat, sample_valid=sample_valid, vl_attention_mask=vl_attention_mask
            )
            logs["consistency_loss"] = consist

        return self.loss_weight * loss, logs

    def consistency_loss(
        self,
        vl_embs: torch.Tensor,
        z0_hat: torch.Tensor,
        sample_valid: Optional[torch.Tensor] = None,
        vl_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Symmetric conditional InfoNCE pulling each predicted future ``z0_hat_i`` toward
        its own context ``c_i`` (masked-mean of valid ``vl_embs`` tokens) and pushing it
        away from the other in-batch contexts ``c_j`` (j != i).

        z       = g_z(mean_{n,h,w} z0_hat)            -> [B, proj], L2-normalized
        c       = g_c(masked_mean_L vl_embs)          -> [B, proj], L2-normalized
        logits  = (z @ c.T) / tau                     -> [B, B]
        L       = 0.5 * (CE(logits, I) + CE(logits.T, I))   (z->c and c->z)
        """
        dtype = self.g_z.weight.dtype
        B = z0_hat.shape[0]

        # Pool z0_hat over spatial/temporal latent dims -> [B, C].
        z_pool = z0_hat.to(dtype).mean(dim=(1, 3, 4))  # mean over (n, h, w)

        # Masked-mean context over VALID VL tokens -> [B, H].
        vl = vl_embs.to(dtype)
        if vl_attention_mask is not None:
            m = vl_attention_mask.to(dtype).unsqueeze(-1)  # [B, L, 1]
            c_pool = (vl * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        else:
            c_pool = vl.mean(dim=1)

        z = F.normalize(self.g_z(z_pool), dim=-1)  # [B, proj]
        c = F.normalize(self.g_c(c_pool), dim=-1)  # [B, proj]

        tau = max(self.consist_temp, 1e-6)
        logits = (z @ c.t()) / tau  # [B, B]
        labels = torch.arange(B, device=logits.device)

        if sample_valid is not None:
            # Mask invalid samples out of the diagonal targets (both directions) so a
            # dropped sample contributes no positive/anchor loss.
            keep = sample_valid.to(torch.bool).to(logits.device)
            if not bool(keep.any()):
                return logits.sum() * 0.0  # gradient-safe zero, no valid anchors
            ignore = labels.masked_fill(~keep, -100)
            loss_zc = F.cross_entropy(logits, ignore, ignore_index=-100)
            loss_cz = F.cross_entropy(logits.t(), ignore, ignore_index=-100)
        else:
            loss_zc = F.cross_entropy(logits, labels)
            loss_cz = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_zc + loss_cz)
