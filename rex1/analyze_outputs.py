"""Compare generated outputs across REX checkpoints (e.g. chat vs code).

Runs the same probe prompts through each model, saves side-by-side completions,
and plots PCA / similarity views of the outputs themselves.

Example:
  python analyze_outputs.py \\
    --model chat:runs/rex1-chat/ckpt_final.pt:config-ouro-stage4-chat.yaml \\
    --model code:runs/rex1-code/ckpt_step15000.pt:config-ouro-stage4-code.yaml \\
    --output-dir benchmark_results/output-analysis-chat-vs-code
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from transformers import AutoTokenizer

from analyze_representations import (
    MODEL_COLORS,
    MODEL_MARKERS,
    PROBE_COLORS,
    PROBE_MARKERS,
    load_probe_bank,
    load_yaml,
    parse_model_arg,
    resolve_device,
)
from model import RexForCausalLM

ROLE_STOP_STRINGS = ("<|system|>", "<|user|>", "<|tool|>", "<|system||>", "< |user|>")


@dataclass
class GeneratedSample:
    label: str
    probe: str
    prompt: str
    completion: str
    num_tokens: int
    hidden_embedding: np.ndarray


def _truncate_at_role_tokens(text: str) -> str:
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
    for stop in ROLE_STOP_STRINGS:
        idx = suffix.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return prefix + suffix[:cut].rstrip()


def _assistant_completion(full_text: str, prompt: str) -> str:
    text = _truncate_at_role_tokens(full_text)
    if text.startswith(prompt):
        return text[len(prompt) :].strip()
    marker = "<|assistant|>\n"
    pos = text.rfind(marker)
    if pos != -1:
        return text[pos + len(marker) :].strip()
    marker = "<|assistant|>"
    pos = text.rfind(marker)
    if pos != -1:
        suffix = text[pos + len(marker) :].lstrip("\n")
        return suffix.strip()
    return text.strip()


@torch.no_grad()
def _embed_generated_tokens(
    *,
    model: RexForCausalLM,
    output_ids: torch.Tensor,
    prompt_len: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    if output_ids.size(1) <= prompt_len:
        return np.zeros(model.cfg.n_embd, dtype=np.float32)

    saved_threshold = model.cfg.early_exit_threshold
    model.cfg.early_exit_threshold = None
    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            hidden_states, _, _ = model._recurrence_forward(
                output_ids,
                normalize=True,
                num_recurrence_steps=None,
                early_exit_threshold=None,
            )
    finally:
        model.cfg.early_exit_threshold = saved_threshold

    gen_hidden = hidden_states[-1][0, prompt_len:, :].float().cpu()
    if gen_hidden.numel() == 0:
        return np.zeros(model.cfg.n_embd, dtype=np.float32)
    return gen_hidden.mean(dim=0).numpy()


@torch.no_grad()
def generate_sample(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    label: str,
    probe: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    no_repeat_ngram_size: int,
    stop_on_role_tokens: bool,
) -> GeneratedSample:
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.size(1)

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_recurrence_steps=None,
            early_exit_threshold=None,
            use_kv_cache=True,
        )

    full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    if stop_on_role_tokens:
        full_text = _truncate_at_role_tokens(full_text)
    completion = _assistant_completion(full_text, prompt)
    embedding = _embed_generated_tokens(
        model=model,
        output_ids=output_ids,
        prompt_len=prompt_len,
        device=device,
        amp_dtype=amp_dtype,
    )
    return GeneratedSample(
        label=label,
        probe=probe,
        prompt=prompt,
        completion=completion,
        num_tokens=max(0, output_ids.size(1) - prompt_len),
        hidden_embedding=embedding,
    )


def _group_by_prompt(samples: list[GeneratedSample]) -> dict[str, list[GeneratedSample]]:
    grouped: dict[str, list[GeneratedSample]] = {}
    for sample in samples:
        key = f"{sample.probe}::{sample.prompt}"
        grouped.setdefault(key, []).append(sample)
    return grouped


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def plot_pca_hidden_embeddings(
    samples: list[GeneratedSample],
    out_path: Path,
    *,
    seed: int,
) -> None:
    matrix = np.stack([s.hidden_embedding for s in samples], axis=0)
    coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    models = sorted({s.label for s in samples})
    probes = sorted({s.probe for s in samples})

    fig, ax = plt.subplots(figsize=(10, 7))
    for model in models:
        for probe in probes:
            mask = [(s.label == model and s.probe == probe) for s in samples]
            if not any(mask):
                continue
            xs = coords[mask, 0]
            ys = coords[mask, 1]
            ax.scatter(
                xs,
                ys,
                s=70,
                alpha=0.75,
                c=MODEL_COLORS.get(model, "#64748b"),
                marker=PROBE_MARKERS.get(probe, "o"),
                edgecolors="white",
                linewidths=0.5,
                label=f"{model} / {probe.replace('_', ' ')}",
            )
    ax.set_title(f"PCA of generated-output hidden states ({len(samples)} completions)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca_text_outputs(
    samples: list[GeneratedSample],
    out_path: Path,
    *,
    seed: int,
) -> None:
    texts = [s.completion or "<empty>" for s in samples]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = tfidf.fit_transform(texts).toarray()
    coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    models = sorted({s.label for s in samples})
    probes = sorted({s.probe for s in samples})

    fig, ax = plt.subplots(figsize=(10, 7))
    for model in models:
        for probe in probes:
            mask = [(s.label == model and s.probe == probe) for s in samples]
            if not any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=70,
                alpha=0.75,
                c=MODEL_COLORS.get(model, "#64748b"),
                marker=PROBE_MARKERS.get(probe, "o"),
                edgecolors="white",
                linewidths=0.5,
                label=f"{model} / {probe.replace('_', ' ')}",
            )
    ax.set_title(f"PCA of generated text (char TF-IDF, {len(samples)} completions)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_output_similarity(
    samples: list[GeneratedSample],
    out_path: Path,
) -> dict[str, float]:
    grouped = _group_by_prompt(samples)
    labels = sorted({s.label for s in samples})
    probes = sorted({s.probe for s in samples})
    if len(labels) < 2:
        return {}

    text_matrix = np.zeros((len(probes), len(labels), len(labels)))
    hidden_matrix = np.zeros((len(probes), len(labels), len(labels)))
    lengths: dict[str, list[int]] = {label: [] for label in labels}

    for sample in samples:
        lengths[sample.label].append(sample.num_tokens)

    for probe_idx, probe in enumerate(probes):
        probe_groups = {k: v for k, v in grouped.items() if k.startswith(f"{probe}::")}
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:
                    text_matrix[probe_idx, i, j] = 1.0
                    hidden_matrix[probe_idx, i, j] = 1.0
                    continue
                sims_text: list[float] = []
                sims_hidden: list[float] = []
                for key, group in probe_groups.items():
                    by_label = {s.label: s for s in group}
                    if a not in by_label or b not in by_label:
                        continue
                    sa, sb = by_label[a], by_label[b]
                    sims_text.append(_text_similarity(sa.completion, sb.completion))
                    va, vb = sa.hidden_embedding, sb.hidden_embedding
                    sims_hidden.append(
                        float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
                    )
                text_matrix[probe_idx, i, j] = float(np.mean(sims_text)) if sims_text else 0.0
                hidden_matrix[probe_idx, i, j] = float(np.mean(sims_hidden)) if sims_hidden else 0.0

    fig, axes = plt.subplots(2, len(probes), figsize=(4.2 * len(probes), 7), squeeze=False)
    for probe_idx, probe in enumerate(probes):
        for row, (matrix, title) in enumerate(
            ((text_matrix, "text similarity"), (hidden_matrix, "hidden cosine"))
        ):
            ax = axes[row, probe_idx]
            im = ax.imshow(matrix[probe_idx], vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_title(f"{probe.replace('_', ' ')} ({title})")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{matrix[probe_idx, i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Paired output similarity for matched prompts", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    chat_idx, code_idx = labels.index("chat") if "chat" in labels else 0, labels.index("code") if "code" in labels else 1
    return {
        "mean_paired_text_similarity": float(np.mean(text_matrix[:, chat_idx, code_idx])),
        "mean_paired_hidden_cosine": float(np.mean(hidden_matrix[:, chat_idx, code_idx])),
        "mean_output_tokens_chat": float(np.mean(lengths.get("chat", [0]))),
        "mean_output_tokens_code": float(np.mean(lengths.get("code", [0]))),
    }


def plot_output_length(samples: list[GeneratedSample], out_path: Path) -> None:
    probes = sorted({s.probe for s in samples})
    labels = sorted({s.label for s in samples})
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(labels))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(probes))
    for offset, label in zip(offsets, labels):
        vals = []
        for probe in probes:
            probe_samples = [s.num_tokens for s in samples if s.label == label and s.probe == probe]
            vals.append(float(np.mean(probe_samples)) if probe_samples else 0.0)
        ax.bar(x + offset, vals, width=width, alpha=0.8, color=MODEL_COLORS.get(label, "#64748b"), label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in probes], fontsize=8)
    ax.set_ylabel("Generated tokens")
    ax.set_title("Output length by probe and model variant")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_paired_displacement(
    samples: list[GeneratedSample],
    out_path: Path,
    *,
    seed: int,
) -> None:
    grouped = _group_by_prompt(samples)
    labels = sorted({s.label for s in samples})
    if len(labels) < 2:
        return

    points: list[tuple[np.ndarray, str]] = []
    for group in grouped.values():
        by_label = {s.label: s for s in group}
        if len(by_label) < 2:
            continue
        anchor = by_label[labels[0]].hidden_embedding
        target = by_label[labels[1]].hidden_embedding
        delta = target - anchor
        probe = next(iter(group)).probe
        points.append((delta, probe))

    if not points:
        return

    matrix = np.stack([p[0] for p in points], axis=0)
    coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(8, 7))
    for (x, y), (_, probe) in zip(coords, points):
        ax.scatter(x, y, s=60, c=PROBE_COLORS.get(probe, "#64748b"), alpha=0.85, edgecolors="white", linewidths=0.4)
        ax.annotate(probe.replace("_", " "), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#cbd5e1", linewidth=0.8)
    ax.axvline(0, color="#cbd5e1", linewidth=0.8)
    ax.set_title(f"Output shift vectors ({labels[1]} − {labels[0]} hidden embeddings)")
    ax.set_xlabel("PC1 of displacement")
    ax.set_ylabel("PC2 of displacement")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_on_text_outputs(
    samples: list[GeneratedSample],
    out_path: Path,
    *,
    seed: int,
) -> dict[str, float]:
    labels = sorted({s.label for s in samples})
    if len(labels) < 2:
        return {}

    texts = [s.completion or "<empty>" for s in samples]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = tfidf.fit_transform(texts).toarray()
    coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    y_true = np.array([labels.index(s.label) for s in samples])
    cluster_ids = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(matrix)

    fig, ax = plt.subplots(figsize=(10, 7))
    cluster_colors = ["#f59e0b", "#06b6d4"]
    for cluster in (0, 1):
        for model in labels:
            mask = (cluster_ids == cluster) & (np.array([s.label for s in samples]) == model)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=70,
                alpha=0.75,
                c=cluster_colors[cluster],
                marker=MODEL_MARKERS.get(model, "o"),
                edgecolors=MODEL_COLORS.get(model, "#111827"),
                linewidths=0.8,
                label=f"cluster {cluster + 1} / {model}",
            )
    ari = float(adjusted_rand_score(y_true, cluster_ids))
    ax.set_title(f"k=2 clusters on generated text | ARI vs model type = {ari:.2f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"text_kmeans_ari": ari}


def write_outputs_markdown(
    samples: list[GeneratedSample],
    out_path: Path,
    metrics: dict[str, float],
) -> None:
    grouped = _group_by_prompt(samples)
    labels = sorted({s.label for s in samples})
    lines = ["# Output comparison", ""]
    if metrics:
        lines.append("## Summary metrics")
        for key, value in metrics.items():
            if "tokens" in key:
                lines.append(f"- {key}: {value:.1f}")
            else:
                lines.append(f"- {key}: {value:.3f}")
        lines.append("")

    for key in sorted(grouped):
        probe, prompt = key.split("::", 1)
        lines.append(f"## {probe}")
        lines.append("")
        lines.append("```")
        lines.append(prompt.rstrip())
        lines.append("```")
        lines.append("")
        by_label = {s.label: s for s in grouped[key]}
        for label in labels:
            sample = by_label.get(label)
            if sample is None:
                continue
            lines.append(f"### {label} ({sample.num_tokens} tokens)")
            lines.append("")
            lines.append("```")
            lines.append(sample.completion or "<empty>")
            lines.append("```")
            lines.append("")
        if len(labels) >= 2 and all(label in by_label for label in labels[:2]):
            a, b = by_label[labels[0]], by_label[labels[1]]
            lines.append(
                f"- text similarity ({labels[0]} vs {labels[1]}): "
                f"{_text_similarity(a.completion, b.completion):.3f}"
            )
            va, vb = a.hidden_embedding, b.hidden_embedding
            lines.append(
                f"- hidden cosine ({labels[0]} vs {labels[1]}): "
                f"{float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)):.3f}"
            )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs_json(samples: list[GeneratedSample], out_path: Path) -> None:
    rows = []
    for sample in samples:
        row = asdict(sample)
        row["hidden_embedding"] = sample.hidden_embedding.tolist()
        rows.append(row)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label:checkpoint:config")
    parser.add_argument("--output-dir", default="benchmark_results/output-analysis")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (recommended)")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--stop-on-role-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = [parse_model_arg(raw) for raw in args.model]
    device = resolve_device(args.device)
    dtype_name = args.dtype.lower()
    amp_dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} and device.type == "cuda" else None
    if dtype_name in {"fp16", "float16"} and device.type == "cuda":
        amp_dtype = torch.float16
    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = load_probe_bank(Path(args.prompts_file) if args.prompts_file else None, expanded=True)

    samples: list[GeneratedSample] = []
    for spec in specs:
        cfg = load_yaml(spec.config)
        tokenizer_name = cfg.get("data", {}).get("tokenizer_name", "HuggingFaceTB/SmolLM2-360M")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        model = RexForCausalLM.from_checkpoint(spec.checkpoint, map_location="cpu").to(device)

        for probe_name, prompt in probes:
            sample = generate_sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                label=spec.label,
                probe=probe_name,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                stop_on_role_tokens=args.stop_on_role_tokens,
            )
            samples.append(sample)
            print(f"[{spec.label}] {probe_name}: {sample.num_tokens} tokens")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics: dict[str, float] = {}
    metrics.update(plot_paired_output_similarity(samples, out_dir / "paired_output_similarity.png"))
    metrics.update(plot_kmeans_on_text_outputs(samples, out_dir / "kmeans_output_text.png", seed=args.seed))
    plot_pca_hidden_embeddings(samples, out_dir / "pca_output_hidden.png", seed=args.seed)
    plot_pca_text_outputs(samples, out_dir / "pca_output_text.png", seed=args.seed)
    plot_output_length(samples, out_dir / "output_length.png")
    plot_paired_displacement(samples, out_dir / "output_shift_vectors.png", seed=args.seed)
    write_outputs_markdown(samples, out_dir / "outputs.md", metrics)
    write_outputs_json(samples, out_dir / "outputs.json")

    print(f"\nWrote output analysis to {out_dir}/")
    print("  outputs.md                   — side-by-side completions")
    print("  pca_output_text.png          — PCA of generated text")
    print("  pca_output_hidden.png        — PCA of output hidden states")
    print("  paired_output_similarity.png — chat vs code per probe")
    print("  output_length.png            — tokens generated by probe")
    print("  output_shift_vectors.png     — code − chat output displacement")
    if metrics:
        print(f"  mean paired text similarity: {metrics.get('mean_paired_text_similarity', float('nan')):.3f}")


if __name__ == "__main__":
    main()
