# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# InternVL3.5 backbone (OpenGVLab), via the HF-native conversion.
#
# USE THE *-HF REPO. The plain `OpenGVLab/InternVL3_5-1B` / `-1B-Instruct` repos ship the custom
# `InternVLChatModel` class: they need trust_remote_code and expose NO `image_token_id` in
# config.json, which is exactly what QwenGR00T's REPA head reads to locate the student's image
# tokens (QwenGR00T.py:238-240). The `-HF` conversion is a native
# `InternVLForConditionalGeneration` and does expose it (151671 for 1B).
#
# Also note `-Instruct` is NOT the recommended checkpoint: per the model card it is the
# CPT+SFT intermediate, while the unsuffixed repo additionally went through CascadeRL and is
# the one OpenGVLab recommends for downstream use.
from typing import Optional

import torch
import torch.nn as nn
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)


class _InternVL_Interface(nn.Module):
    """Wrapper matching the starVLA VLM contract: build_qwenvl_inputs / forward / generate."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "OpenGVLab/InternVL3_5-1B-HF")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = InternVLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.bfloat16
        )
        # max_patches caps InternVL's dynamic tiling. A 224x224 GR1 frame never triggers tiling
        # (it fits one 448 tile), so this is a guard rather than an active knob -- without it a
        # larger input could silently expand to 12 tiles = 3072 image tokens.
        max_patches = int(qwenvl_config.get("max_patches", 1))
        processor = AutoProcessor.from_pretrained(model_id, max_patches=max_patches, min_patches=1)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # QwenGR00T reads model.config.hidden_size for the DiT cross_attention_dim and the REPA
        # student_dim; on InternVL the real width lives under text_config (1024 for 1B).
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        logger.info(
            "InternVL interface: %s | hidden=%d image_token_id=%s attn=%s max_patches=%d",
            model_id,
            self.model.config.hidden_size,
            getattr(self.model.config, "image_token_id", None),
            attn_implementation,
            max_patches,
        )

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """Build batched inputs from raw (images, instructions).

        `images` is [B, [view, ...]]; GR1 sends a single ego view per sample. InternVL's
        processor takes a FLAT image list aligned with the text list (one <image> placeholder
        per image), unlike SmolVLM/Idefics3 which takes a nested per-sample list.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        cot_prompt = None
        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")

        texts, flat_images = [], []
        for imgs, instruction in zip(images, instructions):
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction
            content = [{"type": "image"} for _ in imgs]
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
            flat_images.extend(list(imgs))

        batch_inputs = self.processor(
            text=texts, images=flat_images, return_tensors="pt", padding=True
        )
        if solutions is not None:
            raise NotImplementedError(
                "InternVL interface has no action-token vocabulary, so the FAST/OFT `solutions` "
                "label path does not apply. Use it with QwenGR00T (continuous DiT)."
            )
        return batch_inputs.to(self.model.device)
