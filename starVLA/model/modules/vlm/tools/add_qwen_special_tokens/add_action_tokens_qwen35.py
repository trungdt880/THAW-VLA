# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
#
# Qwen3.5 variant of `add_special_tokens_to_qwen.py`.
#
# Produces the equivalent of StarVLA/Qwen3-VL-4B-Instruct-Action for a Qwen3.5 backbone:
# the tokenizer gains the 2048 <robot_action_i> tokens and the embedding matrix is grown to
# match. NO fine-tuning happens here and none is implied -- the released "-Action" model is
# the stock instruct model plus a wider token codebook.
#
# Differences from the Qwen3-VL script, all forced by the target model or by bugs:
#   1. Qwen3_5ForConditionalGeneration, not Qwen3VLForConditionalGeneration.
#   2. No debugpy.wait_for_client() at import -- the original blocks forever without a debugger.
#   3. attn_implementation is not forced to flash_attention_2 (irrelevant here, and it makes
#      the script fail on machines without flash-attn installed).
#   4. Row/id alignment is FIXED. See below.
#
# The alignment bug, measured on the released Qwen3-VL-4B-Instruct-Action:
#   the tokenizer packs new ids into Qwen's reserved slots starting at len(tokenizer)
#   (151669), but the original script resizes and initialises starting from the *embedding*
#   size (151936). Net effect on the released artifact:
#     ids 151669..151935  (267 tokens) -> kept the pretrained reserved rows, std 0.0071
#     ids 151936..153716 (1781 tokens) -> fresh N(0, 0.02)
#     rows 153717..153983  (267 rows)  -> initialised but unreachable, no token maps to them
#   Here we initialise exactly the rows the new ids occupy, and still size the matrix to
#   old_embed + n_tokens so the 128-row alignment that Qwen relies on is preserved.
import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM


def read_tokens(path: str) -> List[str]:
    seen, out = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--tokens-file", required=True)
    ap.add_argument("--init-strategy", default="normal", choices=["normal", "avg", "zero"])
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--legacy-misaligned",
        action="store_true",
        help="Reproduce the released -Action model's row/id mismatch exactly instead of fixing it.",
    )
    args = ap.parse_args()

    tokens = read_tokens(args.tokens_file)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=getattr(torch, args.dtype), device_map=args.device, trust_remote_code=True
    )

    old_embed_rows = model.get_input_embeddings().weight.shape[0]
    old_tok_len = len(tok)
    to_add = [t for t in tokens if t not in tok.get_vocab()]
    print(f"[info] tokenizer len={old_tok_len}  embed rows={old_embed_rows}  to_add={len(to_add)}")

    added = tok.add_special_tokens({"additional_special_tokens": to_add}) if to_add else 0
    new_tok_len = len(tok)

    # Size the matrix so Qwen's 128-row alignment survives (old_embed + n is a multiple of 128
    # whenever old_embed is), but never smaller than the highest id the tokenizer just handed out.
    target_rows = max(old_embed_rows + added, new_tok_len)
    if target_rows > old_embed_rows:
        model.resize_token_embeddings(target_rows)

    # Rows actually reachable by the tokens we just added.
    init_lo = old_embed_rows if args.legacy_misaligned else old_tok_len
    init_hi = target_rows if args.legacy_misaligned else new_tok_len

    emb = model.get_input_embeddings()
    with torch.no_grad():
        if args.init_strategy == "avg":
            ref = emb.weight[:old_tok_len].mean(dim=0)
            for i in range(init_lo, init_hi):
                emb.weight[i].copy_(ref)
        elif args.init_strategy == "zero":
            for i in range(init_lo, init_hi):
                emb.weight[i].zero_()
        else:
            for i in range(init_lo, init_hi):
                nn.init.normal_(emb.weight[i], mean=0.0, std=args.init_std)
    print(f"[info] initialised rows [{init_lo}, {init_hi}) with {args.init_strategy}")
    print(f"[info] embed rows {old_embed_rows} -> {target_rows}; tokenizer {old_tok_len} -> {new_tok_len}")

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    mapping = {t: tok.convert_tokens_to_ids(t) for t in tokens}
    with open(os.path.join(args.save_dir, "added_custom_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    try:
        proc = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        proc.tokenizer = tok
        proc.save_pretrained(args.save_dir)
    except Exception as e:  # text-only backbones legitimately have no processor
        print(f"[warn] no AutoProcessor saved: {e}")

    rt = AutoTokenizer.from_pretrained(args.save_dir, trust_remote_code=True)
    missing = [t for t in tokens if t not in rt.get_vocab()]
    print("[ok] reload check passed" if not missing else f"[FAIL] missing after reload: {missing[:5]}")
    print(f"[ok] id range {min(mapping.values())}..{max(mapping.values())}")


if __name__ == "__main__":
    main()
