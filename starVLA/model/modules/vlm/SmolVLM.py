# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# SmolVLM (HuggingFaceTB) backbone. Architecturally Idefics3, so it loads through
# Idefics3ForConditionalGeneration.
#
# Why this backbone is usable here at all: it exposes a real `image_token_id` in its config,
# which is what QwenGR00T's REPA head uses to locate the student's image tokens
# (QwenGR00T.py:238-240 -> FastWAMRepaHead). Prismatic/MiniVLA-style backbones inject vision
# features as raw embeddings with no corresponding token id, and would need the REPA selection
# rewritten before they could be distilled into.
from typing import Optional

import torch
import torch.nn as nn
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)


class _SmolVLM_Interface(nn.Module):
    """Wrapper matching the starVLA VLM contract: build_qwenvl_inputs / forward / generate.

    Kept deliberately parallel to `_QWen3_VL_Interface` so QwenGR00T needs no branching.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "HuggingFaceTB/SmolVLM-256M-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        # IMAGE SPLITTING OFF BY DEFAULT -- this is the single most important knob here.
        # SmolVLM's processor otherwise upsamples each frame to 512px and splits it into a
        # 4x4 grid + global view = 17 tiles, giving 1088 image tokens and a ~1145-token
        # sequence for ONE 224x224 GR1 frame. With splitting off: 1 tile, 64 image tokens,
        # ~82-token sequence -- 14x shorter, and 64 happens to match the Cosmos3 teacher
        # cache's 64 tapped image tokens exactly.
        do_split = bool(qwenvl_config.get("do_image_splitting", False))

        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=do_split)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # QwenGR00T reads `model.config.hidden_size` for the DiT cross_attention_dim and the
        # REPA student_dim; on Idefics3 the real width lives under text_config (576 for 256M).
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        logger.info(
            "SmolVLM interface: %s | hidden=%d image_token_id=%s attn=%s image_splitting=%s",
            model_id,
            self.model.config.hidden_size,
            getattr(self.model.config, "image_token_id", None),
            attn_implementation,
            do_split,
        )

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """Build batched model inputs from raw (images, instructions).

        `images` is [B, [PIL/ndarray, ...]] (one list of views per sample); GR1 sends a single
        ego view. Idefics3's processor wants the chat text and the images passed separately,
        unlike Qwen-VL where the images ride inside the message content -- hence the two-step
        apply_chat_template(...) -> processor(text=..., images=...) below.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        cot_prompt = None
        if "CoT_prompt" in self.config.datasets.vla_data:
            cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")

        texts, batch_images = [], []
        for imgs, instruction in zip(images, instructions):
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction
            content = [{"type": "image"} for _ in imgs]
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(msg, add_generation_prompt=True))
            batch_images.append(list(imgs))

        batch_inputs = self.processor(
            text=texts, images=batch_images, return_tensors="pt", padding=True
        )
        if solutions is not None:
            raise NotImplementedError(
                "SmolVLM interface has no action-token vocabulary, so the FAST/OFT "
                "`solutions` label path does not apply. Use it with QwenGR00T (continuous DiT)."
            )
        return batch_inputs.to(self.model.device)
