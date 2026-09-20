"""Multi-depth ladder readout (Gate-0b, minimal version).

Combines the last-K Qwen LM hidden states into a single ``[B, L, H]`` tensor with the
*same shape and dimension* as the current ``hidden_states[-1]`` tap, so it drops into the
existing single-cross-attn action head (and the REPA head) with **no** downstream change.

Design (see ``ACTION_VLM_ARCH_DESIGN.md`` §3 / §5): intermediate LM layers carry
spatial / affordance / grounding signal that the final layer discards. Re-exposing the
last K depths to the action head is the cheapest version of the multi-depth tap that every
panel design converged on (``QwenPI_v3.project_layers`` is the reference idiom).

Module math (per layer k of the last K, each ``h_k`` is ``[B, L, H]``):

    p_k = Linear_k(LayerNorm_k(h_k))                # per-layer projection, H -> H
    w   = softmax(combine_logits)                   # learnable [K] weights, sum to 1
    s   = sum_k  w_k * p_k                           # weighted sum over K layers -> [B, L, H]

The combine logits are initialised so the softmax is biased toward the *last* layer
(``combine_logits[-1]`` large, the rest ~0), and each per-layer ``Linear`` is initialised
to identity. At step 0 this makes ``s`` start near ``h_{-1}`` (the current behaviour), so
training departs gently from the single-depth tap rather than jumping to a random mix.

Zero extra feature dim (output is ``[B, L, H]``); params ~ ``K * (H*H + H)`` (for H=1024,
K=4 that is ~4M). Whole module is gated behind ``framework.distill.use_ladder`` and is only
constructed when that flag is true, so ``use_ladder=false`` is byte-identical to current.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LadderReadout(nn.Module):
    """Softmax-weighted multi-depth readout over the last-K Qwen hidden states.

    Args:
        hidden_size: VLM hidden dim ``H`` (e.g. 1024 for the 0.8B student). Input and
            output both live at ``H`` so the readout is a drop-in for ``hidden_states[-1]``.
        k: number of trailing hidden-state layers to combine (``ladder_k``, default 4).
        last_layer_bias: initial logit mass placed on the final layer so step-0 ~= final
            layer. Larger -> closer to the single-depth tap at init.
    """

    def __init__(self, hidden_size: int, k: int = 4, last_layer_bias: float = 4.0):
        super().__init__()
        if int(k) < 1:
            raise ValueError(f"ladder_k must be >= 1, got {k}.")
        self.hidden_size = int(hidden_size)
        self.k = int(k)

        # Per-layer LayerNorm + Linear(H, H), mirroring the QwenPI_v3 project_layers idiom
        # (LayerNorm then Linear), but keeping the output dim == H (no width compression).
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(self.k)])
        self.projs = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.k)]
        )

        # Learnable [K] combine logits; softmax-normalised at forward time. Biased toward
        # the last layer so the step-0 output ~= h_{-1} (current single-depth behaviour).
        combine_logits = torch.zeros(self.k)
        combine_logits[-1] = float(last_layer_bias)
        self.combine_logits = nn.Parameter(combine_logits)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Init each per-layer Linear to identity so step-0 ~= the (LayerNorm'd) hidden."""
        for proj in self.projs:
            nn.init.eye_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, hidden_states: Union[Sequence[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """Combine the last-K hidden states into a single ``[B, L, H]`` tensor.

        Args:
            hidden_states: ordered list/tuple of K tensors, each ``[B, L, H]``, oldest first
                and the final layer last (i.e. ``qwenvl_outputs.hidden_states[-K:]``).

        Returns:
            ``[B, L, H]`` softmax-weighted sum over the K projected layers.
        """
        if len(hidden_states) != self.k:
            raise ValueError(
                f"LadderReadout expected {self.k} hidden states (ladder_k), got {len(hidden_states)}."
            )

        weights = F.softmax(self.combine_logits, dim=0)  # [K], sums to 1
        out = None
        for w_k, h_k, norm, proj in zip(weights, hidden_states, self.norms, self.projs):
            p_k = proj(norm(h_k))  # [B, L, H]
            term = w_k * p_k
            out = term if out is None else out + term
        return out
