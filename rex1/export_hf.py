"""Export a REX training checkpoint as a Hugging Face model folder."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from configuration_rex import RexConfig
from modeling_rex import RexForCausalLM


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_torch_load(path: str | Path, map_location: str | torch.device | None = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint produced by train.py")
    parser.add_argument("--config", default="config-mixed-2-v3.yaml", help="Training YAML used for tokenizer metadata")
    parser.add_argument("--out-dir", required=True, help="Destination Hugging Face model folder")
    parser.add_argument("--model-name", default="REX1", help="Display name for the generated model card")
    return parser


def write_model_card(out_dir: Path, *, model_name: str, checkpoint: str, train_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
    readme = f"""---
library_name: transformers
pipeline_tag: text-generation
tags:
- custom_code
- causal-lm
- rex
---

# {model_name}

REX is a recursive decoder-only Transformer language model. This repository uses custom
Transformers code, so load it with `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(".", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(".")
```

## Checkpoint

Exported from `{checkpoint}`.

## Training Notes

- Tokenizer: `{data_cfg.get("tokenizer_name", "gpt2")}`
- Context length: `{data_cfg.get("block_size", train_cfg.get("max_seq_len", "unknown"))}`
- Training output dir: `{train_cfg.get("out_dir", "unknown")}`

This is a base language model checkpoint and is not instruction-aligned unless noted.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint.get("model_config") or checkpoint.get("config", {}).get("model")
    if model_cfg is None:
        raise ValueError("checkpoint does not contain model_config or config.model")

    tokenizer_name = data_cfg.get("tokenizer_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model_cfg = dict(model_cfg)
    model_cfg["vocab_size"] = int(model_cfg.get("vocab_size") or len(tokenizer))

    hf_config = RexConfig(
        **model_cfg,
        tokenizer_name=tokenizer_name,
        architectures=["RexForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_rex.RexConfig",
            "AutoModelForCausalLM": "modeling_rex.RexForCausalLM",
        },
    )
    model = RexForCausalLM(hf_config)
    model.rex.load_state_dict(checkpoint["model"])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for filename in ["configuration_rex.py", "modeling_rex.py", "model.py", "inference.py", "requirements.txt"]:
        shutil.copy2(Path(__file__).parent / filename, out_dir / filename)
    with open(out_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(checkpoint_path), "step": checkpoint.get("step")}, f, indent=2)
        f.write("\n")
    write_model_card(out_dir, model_name=args.model_name, checkpoint=str(checkpoint_path), train_cfg=train_cfg, data_cfg=data_cfg)
    print(f"exported Hugging Face model to {out_dir}")


if __name__ == "__main__":
    main()
