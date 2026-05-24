"""Standalone radar chart for REX benchmark JSON results.

Reads benchmark.py JSON files or run_benchmark_suite.py suite JSON and
writes a spider chart PNG comparing checkpoints.

Examples:
  python benchmark_charts.py \\
      --inputs benchmark_results/ckpt_step40000-20260522-131732.json \\
               benchmark_results/ckpt_final-20260523-114344.json \\
      --label stage1-40k stage2-final \\
      --output benchmark_results/comparison-radar.png

  python benchmark_charts.py \\
      --inputs benchmark_results/suite-harness_only-20260524-195240.json \\
      --output benchmark_results/chat-harness-radar.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Preferred metric per benchmark axis. Harness metrics win when both exist.
RADAR_AXES: list[dict[str, Any]] = [
    {
        "label": "HellaSwag",
        "candidates": [
            ("harness", "hellaswag", "acc_norm,none"),
            ("native", "hellaswag", "accuracy"),
        ],
    },
    {
        "label": "ARC-Easy",
        "candidates": [
            ("harness", "arc_easy", "acc_norm,none"),
            ("native", "arc_easy", "accuracy"),
        ],
    },
    {
        "label": "ARC-Challenge",
        "candidates": [
            ("harness", "arc_challenge", "acc_norm,none"),
            ("native", "arc_challenge", "accuracy"),
        ],
    },
    {
        "label": "SciQ",
        "candidates": [
            ("harness", "sciq", "acc,none"),
            ("native", "sciq", "accuracy"),
        ],
    },
    {
        "label": "OpenBookQA",
        "candidates": [
            ("harness", "openbookqa", "acc_norm,none"),
            ("native", "openbookqa", "accuracy"),
        ],
    },
    {
        "label": "WinoGrande",
        "candidates": [
            ("harness", "winogrande", "acc,none"),
            ("native", "winogrande", "accuracy"),
        ],
    },
    {
        "label": "PIQA",
        "candidates": [
            ("harness", "piqa", "acc_norm,none"),
            ("native", "piqa", "accuracy"),
        ],
    },
    {
        "label": "BoolQ",
        "candidates": [
            ("harness", "boolq", "acc,none"),
            ("native", "boolq", "accuracy"),
        ],
    },
    {
        "label": "LAMBADA",
        "candidates": [
            ("harness", "lambada_openai", "acc,none"),
            ("native", "lambada", "accuracy"),
        ],
    },
    {
        "label": "MMLU",
        "candidates": [
            ("harness", "mmlu", "acc,none"),
        ],
    },
    {
        "label": "GSM8K",
        "candidates": [
            ("harness", "gsm8k", "exact_match,strict-match"),
        ],
    },
    {
        "label": "IFEval",
        "candidates": [
            ("harness", "ifeval", "prompt_level_strict_acc,none"),
            ("native", "ifeval_lite", "prompt_strict_accuracy"),
        ],
    },
]

SERIES_COLORS = [
    "#6B7280",
    "#F97316",
    "#9333EA",
    "#8B4513",
    "#EAB308",
    "#22C55E",
    "#3B82F6",
    "#EF4444",
]

# Published lm-eval-style reference scores (percent). Sources in model cards / REX report.
# ARC easy/challenge split from published ARC average using a typical ~8pt gap when only average is listed.
BASELINE_MODELS: dict[str, dict[str, Any]] = {
    "smollm2-base": {
        "label": "SmolLM2-360M (base)",
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M",
        "values": {
            "HellaSwag": 54.5,
            "ARC-Easy": 57.0,
            "ARC-Challenge": 49.0,
            "OpenBookQA": 37.4,
            "WinoGrande": 52.5,
            "PIQA": 71.7,
            "MMLU": 35.8,
            "GSM8K": 3.2,
        },
    },
    "smollm2-instruct": {
        "label": "SmolLM2-360M-Instruct",
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
        "values": {
            "HellaSwag": 52.1,
            "ARC-Easy": 48.0,
            "ARC-Challenge": 39.0,
            "OpenBookQA": 37.4,
            "WinoGrande": 52.5,
            "PIQA": 70.8,
            "MMLU": 32.8,
            "GSM8K": 7.43,
            "IFEval": 41.0,
        },
    },
    "pythia-410m": {
        "label": "Pythia-410M (full train)",
        "source": "REX technical report / EleutherAI lm-eval",
        "values": {
            "HellaSwag": 40.6,
            "ARC-Easy": 48.5,
            "OpenBookQA": 29.4,
        },
    },
    "qwen25-0.5b-instruct": {
        "label": "Qwen2.5-0.5B-Instruct",
        "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        "values": {
            "HellaSwag": 48.0,
            "ARC-Easy": 40.0,
            "ARC-Challenge": 35.0,
            "OpenBookQA": 37.4,
            "WinoGrande": 54.1,
            "PIQA": 67.2,
            "MMLU": 31.7,
            "GSM8K": 26.8,
            "IFEval": 31.6,
        },
    },
}

DEFAULT_BASELINES = ("smollm2-instruct", "smollm2-base", "pythia-410m")


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    return {(row["source"], row["task"], row["metric"]): float(row["value"]) for row in rows}


def _rows_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    if run.get("flat_metrics"):
        return run["flat_metrics"]
    rows: list[dict[str, Any]] = []
    for item in run.get("metrics", []):
        rows.append(
            {
                "source": "native",
                "task": item["task"],
                "metric": item["metric"],
                "value": float(item["value"]),
                "kind": item.get("kind"),
            }
        )
    harness = run.get("harness", {})
    for task_name, task_metrics in harness.get("results", {}).items():
        if not isinstance(task_metrics, dict):
            continue
        for metric_name, value in task_metrics.items():
            if metric_name in {"name", "alias", "sample_len", "version", "task", "group"}:
                continue
            if metric_name.endswith("_stderr") or value is None:
                continue
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "source": "harness",
                    "task": task_name,
                    "metric": metric_name,
                    "value": float(value),
                    "kind": "accuracy" if "acc" in metric_name or "exact_match" in metric_name else "score",
                }
            )
    return rows


def _normalize_percent(value: float, kind: str | None) -> float:
    if kind == "accuracy" or value <= 1.0:
        return max(0.0, min(100.0, value * 100.0))
    return max(0.0, min(100.0, value))


def baseline_radar_series(names: list[str]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for name in names:
        if name not in BASELINE_MODELS:
            available = ", ".join(sorted(BASELINE_MODELS))
            raise ValueError(f"Unknown baseline {name!r}. Available: {available}")
        spec = BASELINE_MODELS[name]
        values = {axis["label"]: None for axis in RADAR_AXES}
        values.update(spec["values"])
        series.append(
            {
                "label": spec["label"],
                "values": values,
                "reference": True,
                "source": spec.get("source"),
            }
        )
    return series


def extract_radar_series(run: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
    lookup = _metric_lookup(_rows_from_run(run))
    values: dict[str, float | None] = {}
    for axis in RADAR_AXES:
        score: float | None = None
        for source, task, metric in axis["candidates"]:
            key = (source, task, metric)
            if key not in lookup:
                continue
            raw = lookup[key]
            kind = "accuracy" if metric in {"accuracy", "prompt_strict_accuracy"} or "acc" in metric or "exact_match" in metric else None
            score = _normalize_percent(raw, kind)
            break
        values[axis["label"]] = score
    run_label = label or run.get("label") or Path(str(run.get("checkpoint", "run"))).stem
    return {"label": run_label, "values": values, "reference": False}


def active_axes(all_series: list[dict[str, Any]]) -> list[str]:
    labels = [axis["label"] for axis in RADAR_AXES]
    active: list[str] = []
    for label in labels:
        if any(series["values"].get(label) is not None for series in all_series):
            active.append(label)
    return active


def load_runs_from_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "runs" in payload:
        return payload["runs"]
    if "metrics" in payload or "flat_metrics" in payload:
        return [payload]
    raise ValueError(f"Unrecognized benchmark JSON format: {path}")


def write_radar_png(
    path: str | Path,
    series: list[dict[str, Any]],
    *,
    title: str = "REX Per Benchmark Performance",
    highlight_last: bool = True,
    y_max: float | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    if not series:
        raise ValueError("No benchmark series to plot")

    axes_labels = active_axes(series)
    if len(axes_labels) < 3:
        raise ValueError("Need at least 3 benchmarks with scores to draw a radar chart")

    angles = np.linspace(0, 2 * math.pi, len(axes_labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    numeric_values = [
        value
        for item in series
        for value in item["values"].values()
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    max_value = max(numeric_values)
    radial_max = y_max or max(100.0, math.ceil(max_value / 10.0) * 10.0)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11)
    ax.set_ylim(0, radial_max)
    ax.set_yticks([radial_max * t for t in (0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.set_yticklabels([f"{int(radial_max * t)}" for t in (0.2, 0.4, 0.6, 0.8, 1.0)], color="#6B7280", fontsize=9)
    ax.grid(color="#D1D5DB", alpha=0.8)
    ax.spines["polar"].set_color("#D1D5DB")

    user_indices = [idx for idx, item in enumerate(series) if not item.get("reference")]
    highlight_idx = user_indices[-1] if user_indices and highlight_last else None

    for idx, item in enumerate(series):
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]
        is_reference = bool(item.get("reference"))
        is_highlight = idx == highlight_idx
        values = [item["values"].get(label) for label in axes_labels]
        plot_values = [float("nan") if value is None else value for value in values]
        plot_values = plot_values + plot_values[:1]
        linewidth = 3.2 if is_highlight else 1.6
        linestyle = "--" if is_reference else "-"
        alpha = 0.20 if is_highlight else 0.08 if is_reference else 0.12
        zorder = 10 if is_highlight else 3 if is_reference else 6
        ax.plot(
            angles,
            plot_values,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o",
            markersize=5 if is_highlight else 3,
            label=item["label"],
            zorder=zorder,
        )
        if is_highlight:
            ax.fill(angles, plot_values, color=color, alpha=alpha, zorder=zorder - 1)

        if is_highlight:
            for angle, value in zip(angles[:-1], values):
                if value is None:
                    continue
                ax.annotate(
                    f"{value:.1f}",
                    xy=(angle, value),
                    xytext=(angle, value + radial_max * 0.045),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 1.2,
                        "alpha": 0.95,
                    },
                )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=24)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(3, len(series)),
        frameon=False,
        fontsize=9,
    )
    plt.tight_layout()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def build_radar_series(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [extract_radar_series(run) for run in runs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Benchmark JSON files or suite JSON (supports globs)",
    )
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--title", default="REX Per Benchmark Performance")
    parser.add_argument("--label", nargs="+", default=None, help="Optional labels per input file")
    parser.add_argument("--no-highlight-last", action="store_true")
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Disable published reference model overlays",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baseline keys: smollm2-base, smollm2-instruct, pythia-410m, qwen25-0.5b-instruct",
    )
    return parser


def main() -> None:
    import glob

    args = build_parser().parse_args()
    runs: list[dict[str, Any]] = []
    labels = args.label or []
    input_paths: list[str] = []
    for pattern in args.inputs:
        input_paths.extend(sorted(glob.glob(pattern)))

    if not input_paths:
        raise SystemExit("No input files matched")

    for idx, path in enumerate(input_paths):
        for run in load_runs_from_json(path):
            if labels and idx < len(labels):
                run = dict(run)
                run["label"] = labels[idx]
            runs.append(run)

    series = build_radar_series(runs)
    if not args.no_baselines:
        baseline_names = [name.strip() for name in args.baselines.split(",") if name.strip()]
        series = baseline_radar_series(baseline_names) + series
    if len(input_paths) == 1 and len(runs) > 1:
        stem = Path(input_paths[0]).stem
        output = args.output or str(Path(input_paths[0]).with_name(f"{stem}-radar.png"))
    else:
        output = args.output or "benchmark_results/benchmark-radar.png"

    out_path = write_radar_png(
        output,
        series,
        title=args.title,
        highlight_last=not args.no_highlight_last,
    )
    print(f"radar/png: {out_path}")


if __name__ == "__main__":
    main()
