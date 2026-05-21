"""Run Ouro-comparable benchmarks via lm-eval-harness when installed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from model import RexForCausalLM


DEFAULT_TASKS = "hellaswag,arc_easy,winogrande,sciq,openbookqa,mmlu,gsm8k"


class RexLmEvalAdapter:
    """Minimal lm-eval adapter for RexForCausalLM."""

    def __init__(
        self,
        model: RexForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        amp_dtype: torch.dtype | None,
        num_recurrence_steps: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.amp_dtype = amp_dtype
        self.num_recurrence_steps = num_recurrence_steps or model.cfg.recurrence_steps
        self._rank = 0
        self._world_size = 1

    @classmethod
    def create(
        cls,
        checkpoint: str,
        config_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        num_recurrence_steps: int | None = None,
    ) -> "RexLmEvalAdapter":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        amp_dtype = None
        if dev.type == "cuda":
            amp_dtype = torch.bfloat16 if dtype.lower() in {"bf16", "bfloat16"} else torch.float16
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        tokenizer = AutoTokenizer.from_pretrained(cfg.get("data", {}).get("tokenizer_name", "HuggingFaceTB/SmolLM2-360M"))
        model = RexForCausalLM.from_checkpoint(checkpoint, map_location="cpu").to(dev)
        model.eval()
        return cls(model, tokenizer, dev, amp_dtype, num_recurrence_steps)

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _model_call(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            return self.model(
                input_ids,
                labels=labels,
                num_recurrence_steps=self.num_recurrence_steps,
            )["logits"]

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in requests:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            if not cont_ids:
                results.append((0.0, False))
                continue
            input_ids = (ctx_ids + cont_ids)[-self.model.cfg.max_seq_len :]
            prompt_kept = max(0, len(input_ids) - len(cont_ids))
            labels = [-100] * prompt_kept + input_ids[prompt_kept:]
            input_tensor = torch.tensor([input_ids], device=self.device)
            label_tensor = torch.tensor([labels], device=self.device)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                out = self.model(
                    input_tensor,
                    labels=label_tensor,
                    num_recurrence_steps=self.num_recurrence_steps,
                )
            loss = float(out["loss"].item())
            logits = out["logits"]
            pred_id = int(torch.argmax(logits[0, len(input_ids) - 2]).item()) if len(input_ids) >= 2 else -1
            greedy = pred_id == cont_ids[0]
            ll = -loss * max(1, len(cont_ids))
            results.append((ll, greedy))
        return results

    def loglikelihood_rolling(self, requests: list[tuple[str, str]]) -> list[float]:
        raise NotImplementedError("rolling loglikelihood not implemented for REX yet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config-mixed-2-v3.yaml")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--recurrence-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None, help="Optional example limit per task")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    try:
        from lm_eval import evaluator
    except ImportError as exc:
        raise SystemExit("Install lm-eval: pip install lm-eval") from exc

    adapter = RexLmEvalAdapter.create(
        args.checkpoint,
        args.config,
        num_recurrence_steps=args.recurrence_steps,
    )
    results = evaluator.simple_evaluate(
        model=adapter,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(yaml.safe_dump(results.get("results", {}), sort_keys=False))
    if args.output:
        Path(args.output).write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
