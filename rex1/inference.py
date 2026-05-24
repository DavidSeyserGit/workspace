"""Generate text from a REX checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from model import RexForCausalLM


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested_device)


def resolve_amp_dtype(device: torch.device, dtype_name: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    dtype_name = dtype_name.lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint produced by train.py")
    parser.add_argument("--prompt", required=True, help="Prompt text to continue")
    parser.add_argument("--config", default="config-ouro-stage4-chat.yaml", help="Path to YAML config")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cpu, etc.")
    parser.add_argument("--dtype", default=None, help="Override inference dtype: bfloat16, float16, or float32")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; 0 means greedy")
    parser.add_argument("--top-k", type=int, default=50, help="Limit sampling to top-k tokens; <=0 disables")
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Prevent repeated n-grams of this size; 0 disables",
    )
    parser.add_argument(
        "--recurrence-steps",
        type=int,
        default=None,
        help="Override recurrence depth during generation",
    )
    parser.add_argument(
        "--early-exit-threshold",
        type=float,
        default=None,
        help="Q-exit CDF threshold (0-1); omit to run all recurrence steps",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV-cache accelerated generation",
    )
    parser.add_argument(
        "--stop-on-role-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate generation at <|system|>, <|user|>, or <|tool|> (default: true)",
    )
    return parser


def _truncate_at_role_tokens(text: str, stop_strings: tuple[str, ...]) -> str:
    """Keep only the first assistant continuation; drop spurious new turns."""
    marker = "<|assistant|>\n"
    pos = text.rfind(marker)
    if pos == -1:
        marker = "<|assistant|>"
        pos = text.rfind(marker)
        if pos == -1:
            return text
        suffix_start = pos + len(marker)
        if suffix_start < len(text) and text[suffix_start] == "\n":
            suffix_start += 1
    else:
        suffix_start = pos + len(marker)

    prefix = text[:suffix_start]
    suffix = text[suffix_start:]
    cut = len(suffix)
    for stop in stop_strings:
        idx = suffix.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return prefix + suffix[:cut].rstrip()


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    device = resolve_device(args.device)
    dtype_name = args.dtype or str(train_cfg.get("dtype", "bfloat16"))
    amp_dtype = resolve_amp_dtype(device, dtype_name)
    tokenizer_name = data_cfg.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    model = RexForCausalLM.from_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            num_recurrence_steps=args.recurrence_steps,
            early_exit_threshold=args.early_exit_threshold,
            use_kv_cache=not args.no_kv_cache,
        )

    print(
        _truncate_at_role_tokens(
            tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True),
            ("<|system|>", "<|user|>", "<|tool|>", "<|system||>", "< |user|>"),
        )
        if args.stop_on_role_tokens
        else tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    )


if __name__ == "__main__":
    main()
