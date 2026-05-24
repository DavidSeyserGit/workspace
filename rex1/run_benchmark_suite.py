"""Run the REX benchmark suite across one or more checkpoints.

Examples:
  python run_benchmark_suite.py --profile standard \\
      --checkpoint runs/rex-ouro-stage1/ckpt_step40000.pt \\
      --config config-ouro-stage1.yaml

  python run_benchmark_suite.py --pipeline ouro

  python run_benchmark_suite.py --profile full \\
      --checkpoints "runs/rex-ouro-stage1/ckpt_step40000.pt" \\
                     "runs/rex-ouro-midtrain/ckpt_final.pt" \\
      --config config-ouro-stage3-midtrain.yaml
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from benchmark import (
    load_yaml,
    make_run_name,
    parse_benchmark_tasks,
    resolve_amp_dtype,
    resolve_device,
    run_model_benchmarks,
    save_comparison,
    save_results,
)
from eval_lm_harness import run_harness_benchmarks, score_continuation_losses
from model import RexForCausalLM


def load_suite(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_profile(suite: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profiles = suite.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile {profile_name!r}. Available: {available}")
    merged = dict(suite.get("defaults", {}))
    merged.update(profiles[profile_name])
    merged["name"] = profile_name
    return merged


def discover_checkpoints(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        path = Path(pattern)
        if path.is_file():
            paths.append(str(path))
        elif path.is_dir():
            paths.extend(str(p) for p in sorted(path.glob("ckpt_*.pt")))
        else:
            raise FileNotFoundError(f"No checkpoint matched: {pattern}")
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def resolve_entries(args: argparse.Namespace, suite: dict[str, Any]) -> list[dict[str, Any]]:
    if args.pipeline:
        pipelines = suite.get("pipelines", {})
        if args.pipeline not in pipelines:
            available = ", ".join(sorted(pipelines))
            raise ValueError(f"Unknown pipeline {args.pipeline!r}. Available: {available}")
        pipeline = pipelines[args.pipeline]
        entries = []
        for item in pipeline.get("entries", []):
            checkpoint = item["checkpoint"]
            if not Path(checkpoint).exists():
                print(f"skip missing checkpoint: {checkpoint}")
                continue
            entries.append(
                {
                    "label": item.get("label") or Path(checkpoint).stem,
                    "checkpoint": checkpoint,
                    "config": item["config"],
                    "profile": item.get("profile") or pipeline.get("profile") or args.profile,
                }
            )
        if not entries:
            raise FileNotFoundError(f"No checkpoints found for pipeline {args.pipeline!r}")
        return entries

    if args.checkpoint or args.checkpoints:
        patterns = list(args.checkpoint or []) + list(args.checkpoints or [])
        checkpoints = discover_checkpoints(patterns)
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        if args.config is None:
            raise ValueError("--config is required when using --checkpoint/--checkpoints")
        return [
            {
                "label": Path(checkpoint).stem,
                "checkpoint": checkpoint,
                "config": args.config,
                "profile": args.profile,
            }
            for checkpoint in checkpoints
        ]

    raise ValueError("Provide --checkpoint, --checkpoints, or --pipeline")


HARNESS_META_KEYS = frozenset({"name", "alias", "sample_len", "version", "task", "group", "task_name"})


def _is_harness_metric_value(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def flatten_metrics(results: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if source == "native":
        for item in results.get("metrics", []):
            rows.append(
                {
                    "source": "native",
                    "task": item["task"],
                    "metric": item["metric"],
                    "value": float(item["value"]),
                    "samples": item.get("samples"),
                    "kind": item.get("kind"),
                    "higher_is_better": item.get("higher_is_better"),
                }
            )
    else:
        for task_name, task_metrics in results.get("results", {}).items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_name, value in task_metrics.items():
                if metric_name in HARNESS_META_KEYS or metric_name.endswith("_stderr"):
                    continue
                if not _is_harness_metric_value(value):
                    continue
                rows.append(
                    {
                        "source": "harness",
                        "task": task_name,
                        "metric": metric_name,
                        "value": float(value),
                        "samples": task_metrics.get("sample_len"),
                        "kind": "accuracy" if "acc" in metric_name or "exact_match" in metric_name else "score",
                        "higher_is_better": True,
                    }
                )
    return rows


def write_markdown_summary(path: Path, runs: list[dict[str, Any]]) -> None:
    key_metrics = [
        ("native", "val", "perplexity"),
        ("native", "wikitext2", "perplexity"),
        ("native", "hellaswag", "accuracy"),
        ("native", "arc_easy", "accuracy"),
        ("native", "arc_challenge", "accuracy"),
        ("native", "sciq", "accuracy"),
        ("native", "openbookqa", "accuracy"),
        ("native", "winogrande", "accuracy"),
        ("native", "piqa", "accuracy"),
        ("native", "boolq", "accuracy"),
        ("native", "lambada", "accuracy"),
        ("native", "ifeval_lite", "prompt_strict_accuracy"),
        ("harness", "hellaswag", "acc_norm,none"),
        ("harness", "arc_easy", "acc_norm,none"),
        ("harness", "mmlu", "acc,none"),
        ("harness", "gsm8k", "exact_match,strict-match"),
        ("harness", "ifeval", "prompt_level_strict_acc,none"),
    ]

    lines = [
        "# REX Benchmark Suite Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Checkpoint | " + " | ".join(f"{src}:{task}/{metric}" for src, task, metric in key_metrics) + " |",
        "| --- | " + " | ".join("---" for _ in key_metrics) + " |",
    ]
    for run in runs:
        label = run.get("label") or Path(run["checkpoint"]).stem
        values = []
        metric_rows = run.get("flat_metrics", [])
        for source, task, metric in key_metrics:
            match = next(
                (
                    row
                    for row in metric_rows
                    if row["source"] == source and row["task"] == task and row["metric"] == metric
                ),
                None,
            )
            if match is None:
                values.append("—")
            elif match["kind"] == "accuracy":
                values.append(f"{match['value'] * 100:.1f}%")
            elif match["metric"] == "perplexity":
                values.append(f"{match['value']:.2f}")
            else:
                values.append(f"{match['value']:.4f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    skipped = []
    for run in runs:
        for item in run.get("native", {}).get("skipped", []):
            skipped.append(f"- {run.get('label')}: native/{item['task']} — {item['error']}")
        for item in run.get("harness", {}).get("skipped", []):
            skipped.append(f"- {run.get('label')}: harness/{item['task']} — {item['error']}")
    if skipped:
        lines.extend(["", "## Skipped tasks", ""] + skipped)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_summary(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = ["label", "checkpoint", "profile", "source", "task", "metric", "value", "samples"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            label = run.get("label") or Path(run["checkpoint"]).stem
            for row in run.get("flat_metrics", []):
                writer.writerow(
                    {
                        "label": label,
                        "checkpoint": run["checkpoint"],
                        "profile": run.get("profile"),
                        "source": row["source"],
                        "task": row["task"],
                        "metric": row["metric"],
                        "value": row["value"],
                        "samples": row.get("samples"),
                    }
                )


def run_entry(
    *,
    entry: dict[str, Any],
    profile: dict[str, Any],
    suite_defaults: dict[str, Any],
    device: str,
    dtype_override: str | None,
    output_dir: str,
    save_individual: bool,
) -> dict[str, Any]:
    cfg = load_yaml(entry["config"])
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    dev = resolve_device(device)
    dtype_name = dtype_override or profile.get("dtype") or suite_defaults.get("dtype") or str(train_cfg.get("dtype", "bfloat16"))
    amp_dtype = resolve_amp_dtype(dev, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.get("tokenizer_name", "HuggingFaceTB/SmolLM2-360M"), use_fast=True)

    native_tasks = parse_benchmark_tasks(profile.get("native_tasks", []))
    harness_tasks = [str(task) for task in profile.get("harness_tasks", []) if str(task).strip()]
    val_batches = int(profile.get("val_batches", suite_defaults.get("val_batches", 20)))
    batch_size = int(profile.get("batch_size", suite_defaults.get("batch_size", 2)))
    mc_examples = int(profile.get("mc_examples", suite_defaults.get("mc_examples", 200)))
    recurrence_steps = profile.get("recurrence_steps", suite_defaults.get("recurrence_steps"))
    recurrence_steps = int(recurrence_steps) if recurrence_steps is not None else None
    early_exit_threshold = profile.get("early_exit_threshold", suite_defaults.get("early_exit_threshold"))
    if early_exit_threshold is not None:
        early_exit_threshold = float(early_exit_threshold)

    label = entry.get("label") or Path(entry["checkpoint"]).stem
    print(f"\n=== {label} ===")
    print(f"checkpoint: {entry['checkpoint']}")
    print(f"config: {entry['config']}")
    print(f"profile: {profile['name']}")

    model = RexForCausalLM.from_checkpoint(entry["checkpoint"], map_location="cpu").to(dev)
    native_results: dict[str, Any] = {"metrics": [], "skipped": []}
    if native_tasks:
        native_results = run_model_benchmarks(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            device=dev,
            amp_dtype=amp_dtype,
            tasks=native_tasks,
            val_batches=val_batches,
            batch_size=batch_size,
            mc_examples=mc_examples,
            recurrence_steps=recurrence_steps,
            early_exit_threshold=early_exit_threshold,
            checkpoint_label=entry["checkpoint"],
            quiet=False,
        )
        native_results["config"] = entry["config"]
        native_results["profile"] = profile["name"]
        if save_individual:
            run_name = make_run_name(entry["checkpoint"])
            json_path, image_path = save_results(native_results, output_dir, run_name)
            print(f"results/json: {json_path}")
            print(f"results/image: {image_path}")

    harness_results: dict[str, Any] = {"results": {}, "skipped": []}
    if harness_tasks:
        harness_batch_size = int(profile.get("harness_batch_size", suite_defaults.get("harness_batch_size", 64)))
        parallel_gpus = int(profile.get("parallel_gpus", suite_defaults.get("parallel_gpus", 2)))
        if parallel_gpus >= 2:
            del model
            if dev.type == "cuda":
                import torch

                torch.cuda.empty_cache()
            model = None
        harness_results = run_harness_benchmarks(
            model=model,
            tokenizer=tokenizer,
            device=dev,
            amp_dtype=amp_dtype,
            tasks=harness_tasks,
            num_fewshot=int(profile.get("harness_num_fewshot", suite_defaults.get("harness_num_fewshot", 0))),
            batch_size=harness_batch_size,
            limit=profile.get("harness_limit", suite_defaults.get("harness_limit")),
            recurrence_steps=recurrence_steps,
            checkpoint=entry["checkpoint"],
            config_path=entry["config"],
            dtype_name=dtype_name,
            parallel_gpus=parallel_gpus,
        )

    if model is not None:
        del model
    if dev.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    flat_metrics = flatten_metrics(native_results, source="native") + flatten_metrics(harness_results, source="harness")
    return {
        "label": label,
        "checkpoint": entry["checkpoint"],
        "config": entry["config"],
        "profile": profile["name"],
        "native": native_results,
        "harness": harness_results,
        "flat_metrics": flat_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="benchmark-suite.yaml", help="Benchmark suite YAML")
    parser.add_argument("--profile", default="standard", help="Profile name from the suite")
    parser.add_argument("--pipeline", default=None, help="Named pipeline preset from the suite")
    parser.add_argument("--checkpoint", nargs="+", default=None, help="One or more checkpoint paths")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Checkpoint paths or globs")
    parser.add_argument("--config", default=None, help="Model/data config YAML for --checkpoint mode")
    parser.add_argument("--device", default=None, help="Override device (auto, cuda, cpu)")
    parser.add_argument("--dtype", default=None, help="Override dtype")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--run-name", default=None, help="Optional output filename stem")
    parser.add_argument("--no-save", action="store_true", help="Print only; do not write result files")
    parser.add_argument(
        "--harness-smoke",
        action="store_true",
        help="Run harness with 100 examples per task (shortcut for --harness-limit 100)",
    )
    parser.add_argument(
        "--harness-limit",
        type=int,
        default=None,
        help="Cap harness tasks to N examples each (overrides profile; --harness-smoke = 100)",
    )
    parser.add_argument(
        "--harness-batch-size",
        type=int,
        default=None,
        help="Harness request batch size (default: 64)",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=int,
        default=None,
        help="Run harness tasks in parallel across N GPUs (default: 2 when available)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite = load_suite(args.suite)
    defaults = suite.get("defaults", {})
    entries = resolve_entries(args, suite)

    output_dir = args.output_dir or defaults.get("output_dir", "benchmark_results")
    device = args.device or defaults.get("device", "auto")
    runs: list[dict[str, Any]] = []

    for entry in entries:
        profile_name = entry.get("profile") or args.profile
        profile = resolve_profile(suite, profile_name)
        harness_limit = args.harness_limit
        if args.harness_smoke and harness_limit is None:
            harness_limit = 100
        if harness_limit is not None:
            profile = dict(profile)
            profile["harness_limit"] = harness_limit
            print(f"harness limit: {harness_limit} examples per task")
        if args.harness_batch_size is not None:
            profile = dict(profile)
            profile["harness_batch_size"] = args.harness_batch_size
        if args.parallel_gpus is not None:
            profile = dict(profile)
            profile["parallel_gpus"] = args.parallel_gpus
        runs.append(
            run_entry(
                entry=entry,
                profile=profile,
                suite_defaults=defaults,
                device=device,
                dtype_override=args.dtype,
                output_dir=output_dir,
                save_individual=not args.no_save and len(entries) == 1,
            )
        )

    if args.no_save:
        print(json.dumps(runs, indent=2))
        return

    run_name = args.run_name or f"suite-{args.pipeline or args.profile}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "profile": args.profile,
        "pipeline": args.pipeline,
        "runs": runs,
    }
    json_path = out_dir / f"{run_name}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / f"{run_name}.md"
    csv_path = out_dir / f"{run_name}.csv"
    write_markdown_summary(md_path, runs)
    write_csv_summary(csv_path, runs)
    print(f"\nsuite/json: {json_path}")
    print(f"suite/markdown: {md_path}")
    print(f"suite/csv: {csv_path}")

    if len(runs) > 1:
        native_only = [run["native"] for run in runs if run.get("native", {}).get("metrics")]
        if native_only:
            cmp_json, cmp_image = save_comparison(native_only, output_dir, f"{run_name}-native-comparison")
            print(f"comparison/json: {cmp_json}")
            print(f"comparison/image: {cmp_image}")


if __name__ == "__main__":
    main()
