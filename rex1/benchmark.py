"""Run lightweight benchmarks for a REX checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from data import build_dataloaders
from model import RexForCausalLM

try:
    from eval_lm_harness import score_continuation_losses
except ImportError:  # pragma: no cover
    score_continuation_losses = None


_EVAL_RECURRENCE_STEPS: int | None = None
_EVAL_EARLY_EXIT_SET: bool = False
_EVAL_EARLY_EXIT_THRESHOLD: float | None = None
_BENCHMARK_QUIET: bool = False


def _model_forward_kwargs(**kwargs: Any) -> dict[str, Any]:
    if _EVAL_EARLY_EXIT_SET:
        kwargs["early_exit_threshold"] = _EVAL_EARLY_EXIT_THRESHOLD
    return kwargs


def _progress(iterable=None, **kwargs):
    if _BENCHMARK_QUIET:
        kwargs["disable"] = True
    return tqdm(iterable, **kwargs)


def parse_benchmark_tasks(tasks: str | list[str]) -> set[str]:
    if isinstance(tasks, list):
        return {task.strip() for task in tasks if str(task).strip()}
    return {task.strip() for task in tasks.split(",") if task.strip()}


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


def add_metric(
    results: dict[str, Any],
    *,
    task: str,
    metric: str,
    value: float,
    samples: int | None,
    kind: str,
    higher_is_better: bool,
) -> None:
    results["metrics"].append(
        {
            "task": task,
            "metric": metric,
            "value": value,
            "samples": samples,
            "kind": kind,
            "higher_is_better": higher_is_better,
        }
    )


def add_skip(results: dict[str, Any], task: str, error: Exception) -> None:
    results["skipped"].append({"task": task, "error": str(error)})


def make_run_name(checkpoint: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{Path(checkpoint).stem}-{timestamp}"


def write_svg_chart(path: Path, results: dict[str, Any]) -> None:
    metrics = results["metrics"]
    width = 1200
    row_height = 34
    header_height = 78
    height = max(220, header_height + row_height * len(metrics) + 40)
    label_width = 280
    bar_width = 760
    rows = []
    for i, item in enumerate(metrics):
        y = header_height + i * row_height
        value = float(item["value"])
        if item["kind"] == "accuracy":
            scaled = max(0.0, min(1.0, value))
            label = f"{value * 100:.1f}%"
        elif item["metric"] == "perplexity":
            scaled = max(0.0, min(1.0, value / 500.0))
            label = f"{value:.2f}"
        else:
            scaled = max(0.0, min(1.0, value / 10.0))
            label = f"{value:.4f}"
        fill = "#2563eb" if item["higher_is_better"] else "#7c3aed"
        rows.append(
            f"""
  <text x="24" y="{y + 21}" class="label">{escape(item['task'])}/{escape(item['metric'])}</text>
  <rect x="{label_width}" y="{y + 5}" width="{bar_width}" height="22" rx="4" fill="#e5e7eb"/>
  <rect x="{label_width}" y="{y + 5}" width="{bar_width * scaled:.1f}" height="22" rx="4" fill="{fill}"/>
  <text x="{label_width + bar_width + 20}" y="{y + 21}" class="value">{label}</text>"""
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px sans-serif; fill: #111827; }}
    .subtitle {{ font: 14px sans-serif; fill: #4b5563; }}
    .label {{ font: 14px sans-serif; fill: #111827; }}
    .value {{ font: 700 14px sans-serif; fill: #111827; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="32" class="title">REX Benchmark Results</text>
  <text x="24" y="56" class="subtitle">{escape(results['checkpoint'])}</text>
  {''.join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_png_chart(path: Path, results: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    metrics = results["metrics"]
    labels = [f"{item['task']}/{item['metric']}" for item in metrics]
    values = [float(item["value"]) * 100 if item["kind"] == "accuracy" else float(item["value"]) for item in metrics]
    colors = ["#2563eb" if item["higher_is_better"] else "#7c3aed" for item in metrics]
    fig_height = max(4.0, 0.42 * len(metrics) + 1.6)
    _, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(f"REX Benchmark Results\n{results['checkpoint']}")
    ax.set_xlabel("Accuracy (%) or raw loss/perplexity")
    for idx, (value, item) in enumerate(zip(values, metrics)):
        text = f"{value:.1f}%" if item["kind"] == "accuracy" else f"{value:.2f}"
        ax.text(value, idx, f" {text}", va="center")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_results(results: dict[str, Any], output_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    png_path = out_dir / f"{run_name}.png"
    try:
        write_png_chart(png_path, results)
        image_path = png_path
    except Exception:
        image_path = out_dir / f"{run_name}.svg"
        write_svg_chart(image_path, results)
    return json_path, image_path


def write_comparison_png(path: Path, runs: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    checkpoints = [Path(run["checkpoint"]).stem.replace("ckpt_", "") for run in runs]
    metric_names = sorted(
        {
            item["task"]
            for run in runs
            for item in run["metrics"]
            if item["kind"] == "accuracy" and item["metric"] == "accuracy"
        }
    )
    _, ax = plt.subplots(figsize=(12, 6))
    for metric_name in metric_names:
        values = []
        for run in runs:
            match = next(
                (
                    item
                    for item in run["metrics"]
                    if item["task"] == metric_name and item["metric"] == "accuracy"
                ),
                None,
            )
            values.append(float(match["value"]) * 100 if match else float("nan"))
        ax.plot(checkpoints, values, marker="o", label=metric_name)
    ax.set_title("REX Benchmark Accuracy Trend")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Checkpoint")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_comparison_svg(path: Path, runs: list[dict[str, Any]]) -> None:
    checkpoints = [Path(run["checkpoint"]).stem.replace("ckpt_", "") for run in runs]
    metric_names = sorted(
        {
            item["task"]
            for run in runs
            for item in run["metrics"]
            if item["kind"] == "accuracy" and item["metric"] == "accuracy"
        }
    )
    width = 1200
    row_height = 34
    label_width = 180
    col_width = 110
    height = max(180, 90 + row_height * len(metric_names))
    headers = "".join(
        f'<text x="{label_width + i * col_width}" y="68" class="subtitle">{escape(label)}</text>'
        for i, label in enumerate(checkpoints)
    )
    rows = []
    for row_idx, metric_name in enumerate(metric_names):
        y = 100 + row_idx * row_height
        cells = []
        for col_idx, run in enumerate(runs):
            match = next(
                (
                    item
                    for item in run["metrics"]
                    if item["task"] == metric_name and item["metric"] == "accuracy"
                ),
                None,
            )
            value = float(match["value"]) * 100 if match else float("nan")
            label = "n/a" if math.isnan(value) else f"{value:.1f}%"
            cells.append(f'<text x="{label_width + col_idx * col_width}" y="{y}" class="value">{label}</text>')
        rows.append(f'<text x="24" y="{y}" class="label">{escape(metric_name)}</text>{"".join(cells)}')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 24px sans-serif; fill: #111827; }}
    .subtitle {{ font: 13px sans-serif; fill: #4b5563; }}
    .label {{ font: 14px sans-serif; fill: #111827; }}
    .value {{ font: 700 14px sans-serif; fill: #2563eb; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="32" class="title">REX Benchmark Accuracy Trend</text>
  {headers}
  {''.join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def save_comparison(runs: list[dict[str, Any]], output_dir: str | Path, run_name: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "runs": runs}
    json_path = out_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    png_path = out_dir / f"{run_name}.png"
    try:
        write_comparison_png(png_path, runs)
        image_path = png_path
    except Exception:
        image_path = out_dir / f"{run_name}.svg"
        write_comparison_svg(image_path, runs)
    return json_path, image_path


@torch.no_grad()
def evaluate_val_ppl(
    *,
    model: RexForCausalLM,
    cfg: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int,
    batch_size: int,
) -> tuple[float, float]:
    train_cfg = dict(cfg.get("train", {}))
    train_cfg["batch_size"] = batch_size
    _, val_loader, _ = build_dataloaders(cfg.get("data", {}), train_cfg)
    if val_loader is None:
        raise ValueError("config does not define a validation token file")

    losses = []
    model.eval()
    for i, batch in enumerate(_progress(val_loader, desc="val_ppl", total=max_batches)):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids, labels=labels, **_model_forward_kwargs())
        losses.append(float(out["loss"].item()))

    loss = sum(losses) / max(1, len(losses))
    return loss, math.exp(min(20, loss))


@torch.no_grad()
def evaluate_text_ppl(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int,
) -> tuple[float, float]:
    ds = load_dataset(dataset_name, dataset_config, split=split) if dataset_config else load_dataset(dataset_name, split=split)
    eos_id = tokenizer.eos_token_id
    tokens: list[int] = []
    for row in ds:
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos_id is not None:
            tokens.append(eos_id)
        needed = (max_batches + 1) * model.cfg.max_seq_len
        if len(tokens) >= needed:
            break

    losses = []
    block_size = model.cfg.max_seq_len
    model.eval()
    for start in _progress(range(0, len(tokens) - block_size + 1, block_size), desc=f"{dataset_name}_ppl", total=max_batches):
        if len(losses) >= max_batches:
            break
        input_ids = torch.tensor([tokens[start : start + block_size]], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids, labels=input_ids, **_model_forward_kwargs())
        losses.append(float(out["loss"].item()))

    loss = sum(losses) / max(1, len(losses))
    return loss, math.exp(min(20, loss))


@torch.no_grad()
def continuation_loss(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    num_recurrence_steps: int | None = None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    if not continuation_ids:
        return float("inf")

    max_seq_len = model.cfg.max_seq_len
    input_ids = (prompt_ids + continuation_ids)[-max_seq_len:]
    prompt_tokens_kept = max(0, len(input_ids) - len(continuation_ids))
    labels = [-100] * prompt_tokens_kept + input_ids[prompt_tokens_kept:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(
            input_tensor,
            labels=label_tensor,
            num_recurrence_steps=num_recurrence_steps if num_recurrence_steps is not None else _EVAL_RECURRENCE_STEPS,
            **_model_forward_kwargs(),
        )
        return float(out["loss"].item())


def _continuation_losses(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuations: list[str],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> list[float]:
    if score_continuation_losses is not None:
        return score_continuation_losses(
            model=model,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            prompt=prompt,
            continuations=continuations,
        )
    return [
        continuation_loss(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuation=continuation,
            device=device,
            amp_dtype=amp_dtype,
        )
        for continuation in continuations
    ]


def evaluate_piqa(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("piqa", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="piqa"):
        prompt = f"Question: {row['goal']}\nAnswer:"
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {row['sol1']}", f" {row['sol2']}"],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = int(losses[1] < losses[0])
        correct += int(pred == int(row["label"]))
        total += 1
    return correct / max(1, total)


def evaluate_hellaswag(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("Rowan/hellaswag", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="hellaswag"):
        ctx = row.get("ctx") or ""
        ctx_a = row.get("ctx_a") or ""
        ctx_b = row.get("ctx_b") or ""
        prompt = f"{ctx}{ctx_a} {ctx_b}".strip() if ctx_a or ctx_b else ctx
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {ending}" for ending in row["endings"]],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = min(range(len(losses)), key=losses.__getitem__)
        correct += int(pred == int(row["label"]))
        total += 1
    return correct / max(1, total)


def evaluate_arc(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    config_name: str,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("ai2_arc", config_name, split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc=config_name.lower().replace("-", "_")):
        prompt = f"Question: {row['question']}\nAnswer:"
        labels = row["choices"]["label"]
        choices = row["choices"]["text"]
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {choice}" for choice in choices],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = labels[min(range(len(losses)), key=losses.__getitem__)]
        correct += int(str(pred) == str(row["answerKey"]))
        total += 1
    return correct / max(1, total)


def evaluate_arc_easy(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    return evaluate_arc(
        model=model,
        tokenizer=tokenizer,
        config_name="ARC-Easy",
        device=device,
        amp_dtype=amp_dtype,
        max_examples=max_examples,
    )


def evaluate_arc_challenge(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    return evaluate_arc(
        model=model,
        tokenizer=tokenizer,
        config_name="ARC-Challenge",
        device=device,
        amp_dtype=amp_dtype,
        max_examples=max_examples,
    )


def evaluate_sciq(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("sciq", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="sciq"):
        prompt = f"Question: {row['question']}\nAnswer:"
        choices = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {choice}" for choice in choices],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = min(range(len(losses)), key=losses.__getitem__)
        correct += int(pred == 0)
        total += 1
    return correct / max(1, total)


def evaluate_openbookqa(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("openbookqa", "main", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="openbookqa"):
        prompt = f"Question: {row['question_stem']}\nAnswer:"
        labels = row["choices"]["label"]
        choices = row["choices"]["text"]
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {choice}" for choice in choices],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = labels[min(range(len(losses)), key=losses.__getitem__)]
        correct += int(str(pred) == str(row["answerKey"]))
        total += 1
    return correct / max(1, total)


def evaluate_winogrande(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("winogrande", "winogrande_xl", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="winogrande"):
        prompt = row["sentence"].replace("_", "")
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[f" {row['option1']}", f" {row['option2']}"],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = str(min(range(len(losses)), key=losses.__getitem__) + 1)
        correct += int(pred == str(row["answer"]))
        total += 1
    return correct / max(1, total)


def evaluate_boolq(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("boolq", split=f"validation[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="boolq"):
        prompt = f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer:"
        losses = _continuation_losses(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            continuations=[" no", " yes"],
            device=device,
            amp_dtype=amp_dtype,
        )
        pred = bool(min(range(len(losses)), key=losses.__getitem__))
        correct += int(pred == bool(row["answer"]))
        total += 1
    return correct / max(1, total)


def evaluate_lambada(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
) -> float:
    ds = load_dataset("EleutherAI/lambada_openai", split=f"test[:{max_examples}]")
    correct = 0
    total = 0
    for row in _progress(ds, desc="lambada"):
        text = row["text"].strip()
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        prompt, target = parts
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            next_logits = model(
                input_ids,
                num_recurrence_steps=_EVAL_RECURRENCE_STEPS,
                **_model_forward_kwargs(),
            )["logits"][:, -1, :]
        pred_id = int(torch.argmax(next_logits, dim=-1).item())
        pred = tokenizer.decode([pred_id]).strip()
        correct += int(pred == target.strip())
        total += 1
    return correct / max(1, total)


IFEVAL_LITE_SUPPORTED = {
    "change_case:english_lowercase",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "keywords:existence",
    "keywords:forbidden_words",
    "length_constraints:number_sentences",
    "punctuation:no_comma",
}


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def _bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line))


def _check_ifeval_instruction(instruction_id: str, kwargs: dict[str, Any], response: str) -> bool:
    text = response.strip()
    lowered = text.lower()
    if instruction_id == "punctuation:no_comma":
        return "," not in text
    if instruction_id == "change_case:english_lowercase":
        letters = [char for char in text if char.isalpha()]
        return bool(letters) and text == text.lower()
    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    if instruction_id == "detectable_format:number_bullet_lists":
        wanted = int(kwargs.get("num_bullets") or 0)
        return wanted > 0 and _bullet_count(text) == wanted
    if instruction_id == "length_constraints:number_sentences":
        wanted = int(kwargs.get("num_sentences") or 0)
        relation = str(kwargs.get("relation") or "exactly").lower()
        actual = _sentence_count(text)
        if relation == "at least":
            return actual >= wanted
        if relation == "at most":
            return actual <= wanted
        return actual == wanted
    if instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords") or []
        return all(str(keyword).lower() in lowered for keyword in keywords)
    if instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words") or []
        return all(str(word).lower() not in lowered for word in forbidden)
    return False


def _ifeval_lite_rows(max_examples: int) -> list[dict[str, Any]]:
    rows = []
    for row in load_dataset("google/IFEval", split="train"):
        instruction_ids = list(row["instruction_id_list"])
        if not instruction_ids or any(instruction_id not in IFEVAL_LITE_SUPPORTED for instruction_id in instruction_ids):
            continue
        rows.append(row)
        if len(rows) >= max_examples:
            break
    return rows


@torch.no_grad()
def evaluate_ifeval_lite(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_examples: int,
    max_new_tokens: int = 96,
) -> tuple[float, dict[str, float]]:
    """Small real IFEval subset for inline SFT checkpoint selection."""
    rows = _ifeval_lite_rows(max_examples)
    passed = 0
    per_instruction: dict[str, list[float]] = {}
    model.eval()
    for row in rows:
        prompt = (
            "<|system|>\nYou are REX, a helpful and honest assistant. Follow the user's instructions exactly.\n"
            f"<|user|>\n{row['prompt']}\n<|assistant|>\n"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=None,
                no_repeat_ngram_size=4,
                num_recurrence_steps=_EVAL_RECURRENCE_STEPS,
                early_exit_threshold=None,
                use_kv_cache=False,
            )
        generated = tokenizer.decode(output_ids[0, input_ids.size(1) :].tolist(), skip_special_tokens=True)
        instruction_results = []
        for instruction_id, instruction_kwargs in zip(row["instruction_id_list"], row["kwargs"], strict=False):
            ok = _check_ifeval_instruction(instruction_id, instruction_kwargs or {}, generated)
            instruction_results.append(ok)
            per_instruction.setdefault(instruction_id.replace(":", "/"), []).append(1.0 if ok else 0.0)
        ok = bool(instruction_results) and all(instruction_results)
        passed += int(ok)
    per_instruction_scores = {
        name: sum(values) / max(1, len(values)) for name, values in per_instruction.items()
    }
    return passed / max(1, len(rows)), per_instruction_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One or more checkpoints produced by train.py")
    parser.add_argument("--config", default="config-ouro-stage1.yaml", help="Path to YAML config")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, cpu, etc.")
    parser.add_argument("--dtype", default=None, help="Override dtype: bfloat16, float16, or float32")
    parser.add_argument(
        "--tasks",
        default="val_ppl,wikitext2_ppl,arc_easy,arc_challenge,hellaswag,sciq,openbookqa,winogrande,boolq,lambada",
        help="Comma-separated tasks",
    )
    parser.add_argument("--val-batches", type=int, default=20, help="Validation batches for perplexity")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for validation perplexity")
    parser.add_argument("--mc-examples", type=int, default=100, help="Examples per classification-style task")
    parser.add_argument(
        "--recurrence-steps",
        type=int,
        default=None,
        help="Recurrence depth at eval (default: model config recurrence_steps)",
    )
    parser.add_argument(
        "--recurrence-sweep",
        action="store_true",
        help="Evaluate MC tasks at each recurrence depth T=1..T_max",
    )
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for JSON results and chart image")
    parser.add_argument("--run-name", default=None, help="Optional output filename stem")
    parser.add_argument("--no-save", action="store_true", help="Print results without writing JSON or image files")
    return parser


def benchmark_wandb_payload(results: dict[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for item in results.get("metrics", []):
        payload[f"bench/{item['task']}/{item['metric']}"] = float(item["value"])
    return payload


def save_benchmark_snapshot(results: dict[str, Any], output_dir: str | Path, step: int) -> Path:
    bench_dir = Path(output_dir) / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    json_path = bench_dir / f"step{step:07d}.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return json_path


def run_model_benchmarks(
    *,
    model: RexForCausalLM,
    cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    tasks: set[str],
    val_batches: int = 20,
    batch_size: int = 2,
    mc_examples: int = 100,
    recurrence_steps: int | None = None,
    early_exit_threshold: float | None | str = "inherit",
    checkpoint_label: str = "live-model",
    quiet: bool = False,
) -> dict[str, Any]:
    global _EVAL_RECURRENCE_STEPS, _EVAL_EARLY_EXIT_SET, _EVAL_EARLY_EXIT_THRESHOLD, _BENCHMARK_QUIET
    previous_quiet = _BENCHMARK_QUIET
    _BENCHMARK_QUIET = quiet
    _EVAL_RECURRENCE_STEPS = recurrence_steps or model.cfg.recurrence_steps
    if early_exit_threshold == "inherit":
        _EVAL_EARLY_EXIT_SET = False
    else:
        _EVAL_EARLY_EXIT_SET = True
        _EVAL_EARLY_EXIT_THRESHOLD = early_exit_threshold
    was_training = model.training
    model.eval()
    results: dict[str, Any] = {
        "checkpoint": checkpoint_label,
        "device": str(device),
        "recurrence_steps": _EVAL_RECURRENCE_STEPS,
        "early_exit_threshold": model.cfg.early_exit_threshold if early_exit_threshold == "inherit" else early_exit_threshold,
        "tasks": sorted(tasks),
        "val_batches": val_batches,
        "batch_size": batch_size,
        "mc_examples": mc_examples,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": [],
        "skipped": [],
    }
    try:
        if "val_ppl" in tasks:
            loss, ppl = evaluate_val_ppl(
                model=model,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=val_batches,
                batch_size=batch_size,
            )
            print(f"val/loss: {loss:.4f}")
            print(f"val/perplexity: {ppl:.2f}")
            add_metric(results, task="val", metric="loss", value=loss, samples=val_batches, kind="loss", higher_is_better=False)
            add_metric(results, task="val", metric="perplexity", value=ppl, samples=val_batches, kind="perplexity", higher_is_better=False)
        if "wikitext2_ppl" in tasks:
            try:
                loss, ppl = evaluate_text_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name="wikitext",
                    dataset_config="wikitext-2-raw-v1",
                    split="test",
                    text_column="text",
                    device=device,
                    amp_dtype=amp_dtype,
                    max_batches=val_batches,
                )
                print(f"wikitext2/loss: {loss:.4f}")
                print(f"wikitext2/perplexity: {ppl:.2f}")
                add_metric(results, task="wikitext2", metric="loss", value=loss, samples=val_batches, kind="loss", higher_is_better=False)
                add_metric(results, task="wikitext2", metric="perplexity", value=ppl, samples=val_batches, kind="perplexity", higher_is_better=False)
            except Exception as exc:
                print(f"wikitext2/skipped: {exc}")
                add_skip(results, "wikitext2", exc)
        if "wikitext103_ppl" in tasks:
            try:
                loss, ppl = evaluate_text_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name="wikitext",
                    dataset_config="wikitext-103-raw-v1",
                    split="test",
                    text_column="text",
                    device=device,
                    amp_dtype=amp_dtype,
                    max_batches=val_batches,
                )
                print(f"wikitext103/loss: {loss:.4f}")
                print(f"wikitext103/perplexity: {ppl:.2f}")
                add_metric(results, task="wikitext103", metric="loss", value=loss, samples=val_batches, kind="loss", higher_is_better=False)
                add_metric(results, task="wikitext103", metric="perplexity", value=ppl, samples=val_batches, kind="perplexity", higher_is_better=False)
            except Exception as exc:
                print(f"wikitext103/skipped: {exc}")
                add_skip(results, "wikitext103", exc)
        if "piqa" in tasks:
            try:
                acc = evaluate_piqa(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"piqa/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="piqa", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"piqa/skipped: {exc}")
                add_skip(results, "piqa", exc)
        if "hellaswag" in tasks:
            try:
                acc = evaluate_hellaswag(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"hellaswag/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="hellaswag", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"hellaswag/skipped: {exc}")
                add_skip(results, "hellaswag", exc)
        if "arc_easy" in tasks:
            try:
                acc = evaluate_arc_easy(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"arc_easy/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="arc_easy", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"arc_easy/skipped: {exc}")
                add_skip(results, "arc_easy", exc)
        if "arc_challenge" in tasks:
            try:
                acc = evaluate_arc_challenge(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"arc_challenge/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="arc_challenge", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"arc_challenge/skipped: {exc}")
                add_skip(results, "arc_challenge", exc)
        if "sciq" in tasks:
            try:
                acc = evaluate_sciq(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"sciq/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="sciq", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"sciq/skipped: {exc}")
                add_skip(results, "sciq", exc)
        if "openbookqa" in tasks:
            try:
                acc = evaluate_openbookqa(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"openbookqa/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="openbookqa", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"openbookqa/skipped: {exc}")
                add_skip(results, "openbookqa", exc)
        if "winogrande" in tasks:
            try:
                acc = evaluate_winogrande(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"winogrande/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="winogrande", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"winogrande/skipped: {exc}")
                add_skip(results, "winogrande", exc)
        if "boolq" in tasks:
            try:
                acc = evaluate_boolq(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"boolq/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="boolq", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"boolq/skipped: {exc}")
                add_skip(results, "boolq", exc)
        if "lambada" in tasks:
            try:
                acc = evaluate_lambada(model=model, tokenizer=tokenizer, device=device, amp_dtype=amp_dtype, max_examples=mc_examples)
                print(f"lambada/accuracy@{mc_examples}: {acc:.3f}")
                add_metric(results, task="lambada", metric="accuracy", value=acc, samples=mc_examples, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"lambada/skipped: {exc}")
                add_skip(results, "lambada", exc)
        if "ifeval_lite" in tasks:
            try:
                score, per_instruction = evaluate_ifeval_lite(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_examples=mc_examples,
                )
                print(f"ifeval_lite/prompt_strict_accuracy@{mc_examples}: {score:.3f}")
                add_metric(results, task="ifeval_lite", metric="prompt_strict_accuracy", value=score, samples=mc_examples, kind="accuracy", higher_is_better=True)
                for name, value in per_instruction.items():
                    add_metric(results, task=f"ifeval_lite/{name}", metric="instruction_accuracy", value=value, samples=None, kind="accuracy", higher_is_better=True)
            except Exception as exc:
                print(f"ifeval_lite/skipped: {exc}")
                add_skip(results, "ifeval_lite", exc)
    finally:
        _BENCHMARK_QUIET = previous_quiet
        _EVAL_EARLY_EXIT_SET = False
        if was_training:
            model.train()
    return results


def run_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint: str,
    cfg: dict[str, Any],
    tokenizer: AutoTokenizer,
    tasks: set[str],
    device: torch.device,
    dtype_name: str,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model = RexForCausalLM.from_checkpoint(checkpoint, map_location="cpu").to(device)
    print(f"checkpoint: {checkpoint}")
    print(f"device: {device}")
    results = run_model_benchmarks(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        tasks=tasks,
        val_batches=args.val_batches,
        batch_size=args.batch_size,
        mc_examples=args.mc_examples,
        recurrence_steps=args.recurrence_steps,
        checkpoint_label=checkpoint,
        quiet=False,
    )
    results["config"] = args.config
    results["dtype"] = dtype_name

    if not args.no_save:
        run_name = make_run_name(checkpoint) if getattr(args, "multi_checkpoint", False) else args.run_name or make_run_name(checkpoint)
        json_path, image_path = save_results(results, args.output_dir, run_name)
        print(f"results/json: {json_path}")
        print(f"results/image: {image_path}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}

    device = resolve_device(args.device)
    dtype_name = args.dtype or str(train_cfg.get("dtype", "bfloat16"))
    amp_dtype = resolve_amp_dtype(device, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer_name", "gpt2"), use_fast=True)
    args.multi_checkpoint = len(args.checkpoint) > 1

    runs = [
        run_checkpoint(
            args=args,
            checkpoint=checkpoint,
            cfg=cfg,
            tokenizer=tokenizer,
            tasks=tasks,
            device=device,
            dtype_name=dtype_name,
            amp_dtype=amp_dtype,
        )
        for checkpoint in args.checkpoint
    ]
    if len(runs) > 1 and not args.no_save:
        run_name = args.run_name or f"checkpoint-comparison-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        json_path, image_path = save_comparison(runs, args.output_dir, f"{run_name}-comparison")
        print(f"comparison/json: {json_path}")
        print(f"comparison/image: {image_path}")


if __name__ == "__main__":
    main()
