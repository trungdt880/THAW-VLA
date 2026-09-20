# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import os
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        # --- LoRA-on-the-LM (opt-in; default OFF -> byte-identical to current C1) ---
        # When `framework.use_lora` is true we freeze the Qwen language-model base weights
        # and adapt them with a low-rank LoRA delta (peft). This preserves general VLM
        # capability (base recoverable by disabling adapters) while letting actions adapt
        # toward full-FT accuracy. ONLY the LM is wrapped — the vision tower, the merger,
        # the action head and the REPA head all stay full-rank trainable (see below).
        self._lora_applied = False
        self._maybe_apply_lora()

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # --- FastWAM distillation (training-only REPA / Channel-1; dropped at inference) ---
        self.distill_cfg = self.config.framework.get("distill", None)

        # --- Gate-0b: multi-depth ladder readout (opt-in; default OFF) ---
        # When `distill.use_ladder` is true, build a LadderReadout that combines the last-K
        # Qwen hidden states into a single [B, L, H] tensor (same shape/dim as the current
        # `hidden_states[-1]` tap) and feeds BOTH the action head and REPA. When false the
        # module is NOT built and the forward/predict paths use `hidden_states[-1]` exactly
        # as before -> byte-identical to current behaviour.
        self.ladder = None
        self.ladder_k = int(self.distill_cfg.get("ladder_k", 4)) if self.distill_cfg is not None else 4
        if self.distill_cfg is not None and bool(self.distill_cfg.get("use_ladder", False)):
            from starVLA.model.modules.distill import LadderReadout

            self.ladder = LadderReadout(
                hidden_size=int(self.qwen_vl_interface.model.config.hidden_size),
                k=self.ladder_k,
            )
            logger.info(
                "Gate-0b ladder readout enabled: K=%d hidden_size=%d (feeds action head + REPA)",
                self.ladder_k,
                self.ladder.hidden_size,
            )

        self.repa_head = None
        self._repa_image_token_id = None
        # distill.repa_student_layer: which VLM layer the REPA loss aligns.
        # None/-1 -> final layer (previous behaviour, unchanged). An int L taps
        # hidden_states[L]. REPA literature reports intermediate depths align better than the
        # last layer, which is what this knob exists to test.
        _rsl = self.distill_cfg.get("repa_student_layer", -1) if self.distill_cfg is not None else -1
        self._repa_student_layer = None if _rsl in (-1, None) else int(_rsl)
        logger.info(
            "REPA student tap layer: %s (action head always reads the final layer)",
            "final (-1)" if self._repa_student_layer is None else f"hidden_states[{self._repa_student_layer}]",
        )
        logger.info(
            "REPA student tap layer: %s (action head always reads the final layer)",
            "final (-1)" if self._repa_student_layer is None else f"hidden_states[{self._repa_student_layer}]",
        )
        # ONLINE V-JEPA2-AC teacher (in-graph; replaces the offline cache). Frozen.
        self.online_repa_teacher = None
        if self.distill_cfg is not None and bool(self.distill_cfg.get("use_repa", False)):
            from starVLA.model.modules.distill import FastWAMRepaHead

            _online = str(self.distill_cfg.get("online_teacher", "") or "").lower()
            # The teacher is TRAINING-ONLY (REPA aux); dropped at inference. Skip building it when
            # serving (eval policy server sets STARVLA_INFERENCE=1) — avoids loading the 16GB V-JEPA
            # ckpt + the app.vjepa_droid import on the eval box, which otherwise hangs server start.
            import os as _os
            _inference = _os.environ.get("STARVLA_INFERENCE", "0") == "1"
            _build_teacher = _online == "vjepa" and not _inference
            if _online == "vjepa":
                _teacher_dim = 1024  # V-JEPA2-AC predictor hidden (so the repa_head sizes correctly)
                if _build_teacher:
                    from starVLA.model.modules.distill import VJepaOnlineTeacher

                    self.online_repa_teacher = VJepaOnlineTeacher(
                        ckpt_path=self.distill_cfg.get("vjepa_ckpt"),
                        vjepa_repo=str(self.distill_cfg.get("vjepa_repo", "/Data2/trungdt/code/vjepa2")),
                        n_cam=int(self.distill_cfg.get("n_cam", 2)),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    _teacher_dim = self.online_repa_teacher.teacher_dim
            else:
                _teacher_dim = int(self.distill_cfg.get("teacher_dim", 3072))

            self.repa_head = FastWAMRepaHead(
                student_dim=int(self.qwen_vl_interface.model.config.hidden_size),
                teacher_dim=_teacher_dim,
                n_cam=int(self.distill_cfg.get("n_cam", 2)),
                mode=str(self.distill_cfg.get("repa_mode", "pooled")),
                loss_weight=float(self.distill_cfg.get("lambda_repa", 0.5)),
            )
            self._repa_image_token_id = int(
                getattr(self.qwen_vl_interface.model.config, "image_token_id", 151655)
            )
            logger.info(
                "FastWAM REPA distillation enabled: mode=%s lambda=%.3f student_dim=%d teacher_dim=%d online=%s",
                self.repa_head.mode,
                self.repa_head.loss_weight,
                self.repa_head.student_dim,
                self.repa_head.teacher_dim,
                self.online_repa_teacher is not None,
            )

        # --- Channel 2: FastWAM future-latent flow head (training-only; opt-in) ---
        self.future_head = None
        if self.distill_cfg is not None and bool(self.distill_cfg.get("use_future", False)):
            from starVLA.model.modules.distill.fastwam_future import FastWAMFutureHead

            for k in ("future_grid_h", "future_grid_w", "n_future"):
                if self.distill_cfg.get(k, None) is None:
                    raise ValueError(f"distill.use_future=true requires distill.{k} (match the future cache meta).")
            # --- Tier-1: conditional InfoNCE future-consistency knobs (opt-in) ---
            # When use_future_consistency=false the head builds no projection heads and
            # the consistency term is never computed -> exact current C2 behavior.
            self._use_future_consistency = bool(self.distill_cfg.get("use_future_consistency", False))
            self._lambda_consist = float(self.distill_cfg.get("lambda_consist", 0.1))
            self._consist_warmup_steps = int(self.distill_cfg.get("consist_warmup_steps", 5000))
            self.future_head = FastWAMFutureHead(
                cross_attention_dim=int(self.qwen_vl_interface.model.config.hidden_size),
                grid_h=int(self.distill_cfg.get("future_grid_h")),
                grid_w=int(self.distill_cfg.get("future_grid_w")),
                n_future=int(self.distill_cfg.get("n_future")),
                latent_channels=int(self.distill_cfg.get("latent_channels", 48)),
                patch=int(self.distill_cfg.get("future_patch", 2)),
                num_layers=int(self.distill_cfg.get("future_layers", 6)),
                loss_weight=float(self.distill_cfg.get("lambda_future", 0.5)),
                use_consistency=self._use_future_consistency,
                consist_proj_dim=int(self.distill_cfg.get("consist_proj_dim", 128)),
                consist_temp=float(self.distill_cfg.get("consist_temp", 0.1)),
            )
            # Per-forward step counter for the consistency-loss linear warmup (0 -> 1).
            # No global step is threaded into forward(); the trainer calls forward once
            # per optimizer micro-step, so this buffer tracks training progress closely
            # enough for a linear warmup. Registered as a buffer so it survives ckpt
            # save/load and DDP. Registered ONLY when consistency is on, so a disabled
            # C2 run has an identical state_dict to the current C2 head.
            if self._use_future_consistency:
                self.register_buffer(
                    "_consist_step", torch.zeros((), dtype=torch.long), persistent=True
                )
            logger.info(
                "FastWAM future-latent head enabled: n_future=%d grid=(%d,%d) C=%d lambda=%.3f"
                " consistency=%s lambda_consist=%.3f warmup=%d",
                self.future_head.n,
                self.future_head.h,
                self.future_head.w,
                self.future_head.C,
                self.future_head.loss_weight,
                self._use_future_consistency,
                self._lambda_consist,
                self._consist_warmup_steps,
            )

    # Default LoRA target modules for the Qwen text decoder.
    #
    # Qwen3.5 is a *hybrid* text stack: `full_attention` layers expose the standard
    # {q,k,v,o}_proj (Qwen3_5Attention) while `linear_attention` layers use a
    # GatedDeltaNet mixer (in_proj_*/out_proj) that these names do NOT match — so on
    # the 0.8B (18 linear + 6 full layers) the attention LoRA lands on the 6 full-attn
    # layers and the MLP LoRA ({gate,up,down}_proj) lands on all 24 layers. That is the
    # intended coverage; override `framework.lora_target_modules` to change it.
    _DEFAULT_LORA_TARGET_MODULES = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def _maybe_apply_lora(self) -> None:
        """Optionally wrap the Qwen language-model submodule with peft LoRA.

        Reads knobs from ``framework`` (top level, alongside ``qwenvl`` /
        ``action_model`` / ``distill``):

          - ``use_lora`` (bool, default False) — master switch. When False this
            method returns immediately and NO peft import is taken, so the model is
            byte-identical to the current C1 path.
          - ``lora_r`` (int, default 32)
          - ``lora_alpha`` (int, default 64)
          - ``lora_dropout`` (float, default 0.0)
          - ``lora_target_modules`` (list[str], optional; default the attn+MLP
            projections in ``_DEFAULT_LORA_TARGET_MODULES``).

        Effect: the LM base weights become frozen (peft sets requires_grad=False on
        everything it does not adapt) and only the LoRA A/B matrices train. The wrapped
        ``language_model`` lives *inside* ``self.qwen_vl_interface``, so its LoRA params
        inherit the ``qwen_vl_interface`` LR group in ``build_param_lr_groups``.

        The vision tower + merger are LEFT UNTOUCHED here, and the action head / REPA
        head are built after this call, so they all stay full-rank trainable. With
        ``use_lora=true`` you must NOT also list the LM under ``trainer.freeze_modules``
        — LoRA already freezes the base (double-freezing the merger would break C1).
        """
        fw = self.config.framework
        if not bool(fw.get("use_lora", False)):
            return

        # Locate the language-model submodule: qwen_vl_interface.model is the HF
        # Qwen*ForConditionalGeneration; .model is the inner Qwen*Model; .language_model
        # is the text decoder (Qwen3_5TextModel for the 0.8B student). Vision lives next
        # to it under .model.visual (+ .visual.merger) and is deliberately not wrapped.
        try:
            inner = self.qwen_vl_interface.model.model
            language_model = inner.language_model
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                "use_lora=true but could not resolve "
                "qwen_vl_interface.model.model.language_model on this VLM backbone; "
                "LoRA-on-the-LM currently supports the Qwen*ForConditionalGeneration "
                "layout. Disable use_lora or extend _maybe_apply_lora()."
            ) from exc

        from peft import LoraConfig, get_peft_model

        target_modules = fw.get("lora_target_modules", None)
        if target_modules:
            target_modules = [str(t) for t in target_modules]
        else:
            target_modules = list(self._DEFAULT_LORA_TARGET_MODULES)

        lora_config = LoraConfig(
            r=int(fw.get("lora_r", 32)),
            lora_alpha=int(fw.get("lora_alpha", 64)),
            lora_dropout=float(fw.get("lora_dropout", 0.0)),
            target_modules=target_modules,
            bias="none",
        )
        # Wrap ONLY the LM submodule and re-attach it in place. peft freezes the LM base
        # weights and makes only lora_A/lora_B trainable.
        peft_language_model = get_peft_model(language_model, lora_config)
        inner.language_model = peft_language_model
        self._lora_applied = True

        n_trainable = sum(p.numel() for p in peft_language_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in peft_language_model.parameters())
        logger.info(
            "LoRA-on-the-LM enabled: r=%d alpha=%d dropout=%.3f targets=%s | "
            "LM params total=%.3fM trainable(LoRA)=%.3fM (base frozen). "
            "Merger + action head + REPA head remain full-rank trainable.",
            int(fw.get("lora_r", 32)),
            int(fw.get("lora_alpha", 64)),
            float(fw.get("lora_dropout", 0.0)),
            target_modules,
            n_total / 1e6,
            n_trainable / 1e6,
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            if self.ladder is not None:
                # Gate-0b: combine the last-K Qwen hidden states into a single [B, L, H]
                # tensor (shape/dim preserved) feeding BOTH the action head and the REPA head.
                last_hidden = self.ladder(qwenvl_outputs.hidden_states[-self.ladder_k:])  # [B, L, H]
            else:
                last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            # --- REPA student tap depth (distill.repa_student_layer; default -1 = final) ---
            # The REPA head reads THIS tensor; the action head keeps using `last_hidden`.
            # Keeping them separate matters: they were the same tensor, so changing the REPA
            # tap would also change what the DiT is conditioned on, confounding a depth
            # ablation with an architecture change. hidden_states is (embeddings, layer_1 ...
            # layer_N), so index L means "output of layer L" for L>=1, and -1 is the final layer.
            repa_hidden = last_hidden
            if self._repa_student_layer is not None:
                repa_hidden = qwenvl_outputs.hidden_states[self._repa_student_layer]

        # Step 3.5: FastWAM REPA auxiliary loss (training-only). Computed on the
        # un-repeated batch in float32; folded into action_loss below because the
        # trainer only reads output_dict["action_loss"].
        repa_loss_term = None
        repa_logs = {}
        if (
            self.repa_head is not None
            and self.online_repa_teacher is not None
            and examples[0].get("repa_online_clips", None) is not None
        ):
            # --- ONLINE V-JEPA2-AC teacher: compute the target in-graph (no cache). ---
            # Per-sample clips are list[n_cam] of [3,T,256,256]; stack to a single batched
            # tensor [B, n_cam, 3, T, 256, 256] (T=8 fixed -> uniform shapes). States/actions
            # are [B,T,7]. The frozen teacher runs under no_grad/autocast and returns
            # [B, n_cam, 1024]; we flatten to [B, n_cam*1024] for the (pooled) repa_head.
            clips_per_cam = torch.as_tensor(
                np.stack([np.stack([np.asarray(c, dtype=np.float32) for c in e["repa_online_clips"]], axis=0)
                          for e in examples]),
                device=last_hidden.device,
                dtype=torch.float32,
            )  # [B, n_cam, 3, T, 256, 256]
            states_t = torch.as_tensor(
                np.stack([np.asarray(e["repa_online_states"], dtype=np.float32) for e in examples]),
                device=last_hidden.device,
                dtype=torch.float32,
            )  # [B, T, 7]
            actions_t = torch.as_tensor(
                np.stack([np.asarray(e["repa_online_actions"], dtype=np.float32) for e in examples]),
                device=last_hidden.device,
                dtype=torch.float32,
            )  # [B, T, 7]
            with torch.no_grad():
                teacher_bnc = self.online_repa_teacher(clips_per_cam, states_t, actions_t)  # [B,n_cam,1024]
            B_t = teacher_bnc.shape[0]
            teacher_target = teacher_bnc.reshape(B_t, -1).to(last_hidden.device, torch.float32)  # [B, n_cam*1024]
            sample_valid = torch.as_tensor(
                [bool(e.get("repa_online_valid", True)) for e in examples],
                device=last_hidden.device,
                dtype=torch.bool,
            )
            repa_loss_term, repa_logs = self.repa_head(
                last_hidden=repa_hidden.float(),
                input_ids=qwen_inputs["input_ids"],
                image_token_id=self._repa_image_token_id,
                teacher_target=teacher_target,
                sample_valid=sample_valid,
                grid_hw=None,
            )
        elif self.repa_head is not None and examples[0].get("fastwam_target", None) is None:
            # use_repa is on and a projector exists, but no sample carries a teacher target.
            # The usual cause is a mistyped/absent `fastwam_target_cache`: REPA then silently
            # degrades to a plain baseline while the config still claims distillation.
            if not getattr(self, "_repa_no_target_warned", False):
                self._repa_no_target_warned = True
                print(
                    "[REPA] WARNING: use_repa=true and the projector is built, but this batch has "
                    "NO teacher targets. Check datasets.vla_data.fastwam_target_cache points at a "
                    "real cache -- otherwise this run is an undistilled baseline.",
                    flush=True,
                )
        elif self.repa_head is not None and examples[0].get("fastwam_target", None) is not None:
            teacher_target = torch.as_tensor(
                np.stack([np.asarray(e["fastwam_target"], dtype=np.float32) for e in examples]),
                device=last_hidden.device,
                dtype=torch.float32,
            )  # [B, target_dim]
            sample_valid = torch.as_tensor(
                [bool(e.get("fastwam_valid", True)) for e in examples],
                device=last_hidden.device,
                dtype=torch.bool,
            )
            grid_hw = None
            if self.repa_head.mode == "spatial":
                gh = self.distill_cfg.get("grid_h", None)
                gw = self.distill_cfg.get("grid_w", None)
                grid_hw = (int(gh), int(gw)) if gh and gw else None
            repa_loss_term, repa_logs = self.repa_head(
                last_hidden=repa_hidden.float(),
                input_ids=qwen_inputs["input_ids"],
                image_token_id=self._repa_image_token_id,
                teacher_target=teacher_target,
                sample_valid=sample_valid,
                grid_hw=grid_hw,
            )

        # Step 3.6: FastWAM future-latent flow loss (Channel 2, training-only, opt-in).
        future_loss_term = None
        future_logs = {}
        if self.future_head is not None and examples[0].get("fastwam_future_target", None) is not None:
            fut_flat = torch.as_tensor(
                np.stack([np.asarray(e["fastwam_future_target"], dtype=np.float32) for e in examples]),
                device=last_hidden.device,
                dtype=torch.float32,
            )  # [B, n*C*h*w]
            B = fut_flat.shape[0]
            fut = fut_flat.reshape(
                B, self.future_head.n, self.future_head.C, self.future_head.h, self.future_head.w
            )
            fvalid = torch.as_tensor(
                [bool(e.get("fastwam_future_valid", True)) for e in examples],
                device=last_hidden.device,
                dtype=torch.bool,
            )
            future_loss_term, future_logs = self.future_head.flow_loss(
                last_hidden.float(),
                fut,
                sample_valid=fvalid,
                vl_attention_mask=backbone_attention_mask,
            )

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # DIT_NO_ENCODER_MASK=1 suppresses the cross-attention mask, reproducing the
            # conditioning the RELEASED StarVLA checkpoints were trained under: the Dec-2025 code
            # called `self.action_model(hidden, actions, state)` with no mask at all
            # (f18fbc2 QwenGR00T.py:130, and :179 for predict_action). Masking is the more correct
            # behaviour, but the processor left-pads, so masking vs not changes the DiT's
            # conditioning geometry -- finetuning or evaluating those weights WITH a mask is a
            # train/test mismatch. Default off: current behaviour is unchanged.
            _no_enc_mask = os.environ.get("DIT_NO_ENCODER_MASK", "0") not in ("0", "", "false", "False")
            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=None if _no_enc_mask else backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        # Fold aux distillation losses into action_loss (trainer reads only "action_loss")
        # and surface detached scalars for logging.
        out = {"action_loss": action_loss}
        aux_logs = {}
        if repa_loss_term is not None:
            out["action_loss"] = out["action_loss"] + repa_loss_term.to(action_loss.dtype)
            aux_logs.update(repa_logs)
        if future_loss_term is not None:
            out["action_loss"] = out["action_loss"] + future_loss_term.to(action_loss.dtype)
            # --- Tier-1: conditional InfoNCE future-consistency term (opt-in) ---
            # Folded alongside the flow anchor with its own lambda and a linear warmup.
            # When use_future_consistency=false, future_logs has no "consistency_loss"
            # key, so action_loss is byte-identical to current C2.
            consist_term = future_logs.pop("consistency_loss", None)
            if consist_term is not None:
                if self.training:
                    step = int(self._consist_step.item())
                    self._consist_step += 1
                else:
                    step = int(self._consist_step.item())
                warmup = self._consist_warmup_steps
                w = 1.0 if warmup <= 0 else min(1.0, step / float(warmup))
                weighted_consist = (self._lambda_consist * w) * consist_term
                out["action_loss"] = out["action_loss"] + weighted_consist.to(action_loss.dtype)
                future_logs["consistency_loss"] = consist_term.detach()
                future_logs["consist_warmup_w"] = torch.as_tensor(w, device=action_loss.device)
            aux_logs.update(future_logs)
        for k, v in aux_logs.items():
            out[k] = v
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            if self.ladder is not None:
                # Gate-0b: same multi-depth substitution as forward() — MUST mirror it so
                # train and inference read identical features (a mismatch silently breaks eval).
                last_hidden = self.ladder(qwenvl_outputs.hidden_states[-self.ladder_k:])  # [B, L, H]
            else:
                last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            # See DIT_NO_ENCODER_MASK note in forward(): released checkpoints were trained and
            # evaluated with no cross-attention mask, so serving them with one is a mismatch.
            _no_enc_mask = os.environ.get("DIT_NO_ENCODER_MASK", "0") not in ("0", "", "false", "False")
            pred_actions = self.action_model.predict_action(
                last_hidden, state,
                encoder_attention_mask=None if _no_enc_mask else backbone_attention_mask,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
