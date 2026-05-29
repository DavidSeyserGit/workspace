"""Run Ouro-comparable benchmarks via lm-eval-harness when installed."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from model import RexForCausalLM

try:
    from lm_eval.api.model import LM
except ImportError:  # pragma: no cover - optional dependency at import time
    LM = object  # type: ignore[misc,assignment]


DEFAULT_TASKS = "hellaswag,arc_easy,winogrande,sciq,openbookqa,mmlu,gsm8k"
DEFAULT_HARNESS_BATCH_SIZE = 64
CODE_EVAL_TASKS = frozenset({"humaneval", "mbpp"})


def _needs_code_eval(tasks: list[str]) -> bool:
    return any(task.lower() in CODE_EVAL_TASKS for task in tasks)


def _enable_code_eval() -> None:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")


class RexLmEvalAdapter(LM):
    """lm-eval adapter for RexForCausalLM."""

    def __init__(
        self,
        model: RexForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        amp_dtype: torch.dtype | None,
        num_recurrence_steps: int | None = None,
        batch_size: int = DEFAULT_HARNESS_BATCH_SIZE,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.amp_dtype = amp_dtype
        self.num_recurrence_steps = num_recurrence_steps or model.cfg.recurrence_steps
        self._batch_size = max(1, int(batch_size))

    @classmethod
    def create(
        cls,
        checkpoint: str,
        config_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        num_recurrence_steps: int | None = None,
        batch_size: int = DEFAULT_HARNESS_BATCH_SIZE,
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
        return cls(model, tokenizer, dev, amp_dtype, num_recurrence_steps, batch_size=batch_size)

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "rex-tokenizer")

    def tok_encode(self, string: str, add_special_tokens: bool | None = None) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _score_continuation(self, context: str, continuation: str) -> tuple[float, bool]:
        results = self._score_batch([(context, continuation)])
        return results[0]

    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[tuple[float, bool]]:
        if not pairs:
            return []
        if len(pairs) == 1:
            context, continuation = pairs[0]
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            if not cont_ids:
                return [(0.0, False)]
            input_ids = (ctx_ids + cont_ids)[-self.model.cfg.max_seq_len :]
            prompt_kept = max(0, len(input_ids) - len(cont_ids))
            labels = [-100] * prompt_kept + input_ids[prompt_kept:]
            input_tensor = torch.tensor([input_ids], device=self._device)
            label_tensor = torch.tensor([labels], device=self._device)
            with torch.amp.autocast(
                device_type=self._device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
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
            return [(ll, greedy)]

        max_seq_len = self.model.cfg.max_seq_len
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        seqs: list[list[int]] = []
        prompt_lens: list[int] = []
        cont_ids_list: list[list[int]] = []
        for context, continuation in pairs:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            if not cont_ids:
                seqs.append([pad_id])
                prompt_lens.append(1)
                cont_ids_list.append([])
                continue
            input_ids = (ctx_ids + cont_ids)[-max_seq_len:]
            prompt_len = max(0, len(input_ids) - len(cont_ids))
            seqs.append(input_ids)
            prompt_lens.append(prompt_len)
            cont_ids_list.append(cont_ids)

        batch_max = max(len(seq) for seq in seqs)
        batch_input = torch.full((len(seqs), batch_max), pad_id, dtype=torch.long, device=self._device)
        for i, seq in enumerate(seqs):
            batch_input[i, batch_max - len(seq) :] = torch.tensor(seq, device=self._device)

        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            logits = self.model(
                batch_input,
                num_recurrence_steps=self.num_recurrence_steps,
            )["logits"]

        results: list[tuple[float, bool]] = []
        for i, (prompt_len, cont_ids) in enumerate(zip(prompt_lens, cont_ids_list)):
            if not cont_ids:
                results.append((0.0, False))
                continue
            offset = batch_max - len(seqs[i])
            start = offset + prompt_len
            end = start + len(cont_ids)
            tok_logits = logits[i, start - 1 : end - 1, :]
            tok_targets = batch_input[i, start:end]
            loss = F.cross_entropy(tok_logits, tok_targets, reduction="mean")
            pred_id = int(torch.argmax(tok_logits[0]).item())
            greedy = pred_id == int(tok_targets[0].item())
            ll = -float(loss.item()) * max(1, len(cont_ids))
            results.append((ll, greedy))
        return results

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        pairs = [request.arguments for request in requests]
        results: list[tuple[float, bool]] = []
        batch_size = self._batch_size
        for start in _progress(range(0, len(pairs), batch_size), disable=disable_tqdm, desc="loglikelihood"):
            chunk = pairs[start : start + batch_size]
            results.extend(self._score_batch(chunk))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results: list[float] = []
        block_size = self.model.cfg.max_seq_len
        for request in tqdm(requests, disable=disable_tqdm, desc="loglikelihood_rolling"):
            (text,) = request.arguments
            token_ids = self.tok_encode(text)
            if not token_ids:
                results.append(0.0)
                continue
            total_ll = 0.0
            count = 0
            for start in range(0, len(token_ids), block_size):
                chunk = token_ids[start : start + block_size]
                if len(chunk) < 2:
                    continue
                input_tensor = torch.tensor([chunk], device=self._device)
                with torch.amp.autocast(
                    device_type=self._device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_dtype is not None,
                ):
                    out = self.model(
                        input_tensor,
                        labels=input_tensor,
                        num_recurrence_steps=self.num_recurrence_steps,
                    )
                loss = float(out["loss"].item())
                total_ll += -loss * len(chunk)
                count += len(chunk)
            results.append(total_ll / max(1, count))
        return results

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        results: list[str] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, gen_kwargs = request.arguments
            until = gen_kwargs.get("until", [])
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", 256))
            input_ids = self.tok_encode(context)
            input_tensor = torch.tensor([input_ids[-self.model.cfg.max_seq_len :]], device=self._device)
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_gen_toks,
                temperature=0.0,
                no_repeat_ngram_size=4,
                num_recurrence_steps=self.num_recurrence_steps,
                early_exit_threshold=None,
                use_kv_cache=False,
            )
            generated = self.tok_decode(output_ids[0, input_tensor.size(1) :].tolist())
            for stop_seq in until:
                if stop_seq and stop_seq in generated:
                    generated = generated.split(stop_seq)[0]
            results.append(generated)
        return results


def _progress(iterable, *, disable: bool, desc: str):
    if disable:
        return iterable
    return tqdm(iterable, desc=desc)


@torch.no_grad()
def score_continuation_losses(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    prompt: str,
    continuations: list[str],
    batch_size: int = DEFAULT_HARNESS_BATCH_SIZE,
    num_recurrence_steps: int | None = None,
) -> list[float]:
    """Batched continuation-loss scoring for native MC benchmarks."""
    adapter = RexLmEvalAdapter(
        model,
        tokenizer,
        device,
        amp_dtype,
        num_recurrence_steps=num_recurrence_steps,
        batch_size=batch_size,
    )
    pairs = [(prompt, continuation) for continuation in continuations]
    losses: list[float] = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        for (context, continuation), (ll, _) in zip(chunk, adapter._score_batch(chunk), strict=False):
            cont_len = max(1, len(adapter.tok_encode(continuation)))
            losses.append(-ll / cont_len)
    return losses


def _run_single_harness_task(
    *,
    task: str,
    checkpoint: str,
    config_path: str,
    device: str,
    dtype: str,
    num_fewshot: int,
    batch_size: int,
    limit: float | None,
    recurrence_steps: int | None,
) -> tuple[str, dict[str, Any], str | None]:
    if task.lower() in CODE_EVAL_TASKS:
        _enable_code_eval()

    try:
        from lm_eval import evaluator
    except ImportError as exc:
        return task, {}, f"Install lm-eval: pip install lm-eval ({exc})"

    adapter = RexLmEvalAdapter.create(
        checkpoint,
        config_path,
        device=device,
        dtype=dtype,
        num_recurrence_steps=recurrence_steps,
        batch_size=batch_size,
    )
    payload = evaluator.simple_evaluate(
        model=adapter,
        tasks=[task],
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        confirm_run_unsafe_code=task.lower() in CODE_EVAL_TASKS,
    )
    task_results = payload.get("results", {}) if payload else {}
    return task, task_results, None


def _harness_worker(payload: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    return _run_single_harness_task(**payload)


def run_harness_benchmarks(
    *,
    model: RexForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
    tasks: list[str],
    num_fewshot: int = 0,
    batch_size: int = DEFAULT_HARNESS_BATCH_SIZE,
    limit: float | None = None,
    recurrence_steps: int | None = None,
    checkpoint: str | None = None,
    config_path: str | None = None,
    dtype_name: str = "bfloat16",
    parallel_gpus: int = 1,
) -> dict[str, Any]:
    code_eval_tasks = _needs_code_eval(tasks)
    if code_eval_tasks:
        _enable_code_eval()

    try:
        from lm_eval.api.model import LM as LMEvalModel
    except ImportError as exc:
        return {
            "results": {},
            "skipped": [{"task": "harness", "error": f"Install lm-eval: pip install lm-eval ({exc})"}],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_parallel = parallel_gpus >= 2 and gpu_count >= 2 and checkpoint and config_path
    results: dict[str, Any] = {
        "results": {},
        "skipped": [],
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "batch_size": batch_size,
        "parallel_gpus": parallel_gpus if use_parallel else 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if use_parallel:
        print(f"harness: running {len(tasks)} task(s) across {min(parallel_gpus, gpu_count)} GPU(s), batch_size={batch_size}")
        worker_payloads = [
            {
                "task": task,
                "checkpoint": checkpoint,
                "config_path": config_path,
                "device": f"cuda:{idx % min(parallel_gpus, gpu_count)}",
                "dtype": dtype_name,
                "num_fewshot": num_fewshot,
                "batch_size": batch_size,
                "limit": limit,
                "recurrence_steps": recurrence_steps,
            }
            for idx, task in enumerate(tasks)
        ]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(parallel_gpus, gpu_count), mp_context=ctx) as executor:
            futures = [executor.submit(_harness_worker, payload) for payload in worker_payloads]
            for future in as_completed(futures):
                task, task_results, error = future.result()
                if error:
                    print(f"harness/{task}/skipped: {error}")
                    results["skipped"].append({"task": task, "error": error})
                    continue
                results["results"].update(task_results)
                for metric_name, value in next(iter(task_results.values()), {}).items():
                    if metric_name in {"name", "alias", "sample_len"} or metric_name.endswith("_stderr") or value is None:
                        continue
                    if not isinstance(value, (int, float)):
                        continue
                    print(f"harness/{task}/{metric_name}: {value}")
        return results

    if model is None or tokenizer is None or device is None:
        if not checkpoint or not config_path:
            raise ValueError("run_harness_benchmarks requires model+tokenizer or checkpoint+config_path")
        adapter = RexLmEvalAdapter.create(
            checkpoint,
            config_path,
            device=str(device or "auto"),
            dtype=dtype_name,
            num_recurrence_steps=recurrence_steps,
            batch_size=batch_size,
        )
        model = adapter.model
        tokenizer = adapter.tokenizer
        device = adapter._device
        amp_dtype = adapter.amp_dtype

    adapter = RexLmEvalAdapter(
        model,
        tokenizer,
        device,
        amp_dtype,
        num_recurrence_steps=recurrence_steps,
        batch_size=batch_size,
    )
    if not isinstance(adapter, LMEvalModel):
        return {
            "results": {},
            "skipped": [{"task": "harness", "error": "RexLmEvalAdapter is not a valid lm-eval LM subclass"}],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    from lm_eval import evaluator

    for task in tasks:
        try:
            print(f"harness/{task}: starting (fewshot={num_fewshot}, limit={limit}, batch_size={batch_size})")
            payload = evaluator.simple_evaluate(
                model=adapter,
                tasks=[task],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                confirm_run_unsafe_code=task.lower() in CODE_EVAL_TASKS,
            )
            task_results = payload.get("results", {}) if payload else {}
            results["results"].update(task_results)
            for metric_name, value in next(iter(task_results.values()), {}).items():
                if metric_name in {"name", "alias", "sample_len"} or metric_name.endswith("_stderr") or value is None:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                print(f"harness/{task}/{metric_name}: {value}")
        except Exception as exc:
            print(f"harness/{task}/skipped: {exc}")
            results["skipped"].append({"task": task, "error": str(exc)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config-ouro-stage1.yaml")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--recurrence-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HARNESS_BATCH_SIZE)
    parser.add_argument("--parallel-gpus", type=int, default=2, help="Run harness tasks in parallel across N GPUs")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run with 100 examples per task (shortcut for --limit 100)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit per task")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    limit = args.limit
    if args.smoke and limit is None:
        limit = 100
    results = run_harness_benchmarks(
        checkpoint=args.checkpoint,
        config_path=args.config,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=limit,
        recurrence_steps=args.recurrence_steps,
        parallel_gpus=args.parallel_gpus,
    )
    print(yaml.safe_dump(results.get("results", {}), sort_keys=False))
    if args.output:
        Path(args.output).write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
