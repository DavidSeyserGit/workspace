"""Compare internal representations across REX checkpoints (e.g. chat vs code).

Extracts per-recurrence hidden states and exit-gate profiles, then writes PNG
summaries to an output directory.

Example:
  python analyze_representations.py \\
    --model chat:runs/rex1-chat/ckpt_final.pt:config-ouro-stage4-chat.yaml \\
    --model code:runs/rex1-code/ckpt_step15000.pt:config-ouro-stage4-code.yaml \\
    --output-dir benchmark_results/repr-analysis
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, silhouette_score
from transformers import AutoTokenizer

from model import RexForCausalLM
from ouro_loss import exit_distribution


# Small set for recurrence / exit-gate line plots (readable legends).
CORE_PROBES: list[tuple[str, str]] = [
    ("neutral_fact", "The capital of France is Paris. The capital of Germany is"),
    (
        "chat_qa",
        "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is 2+2?\n<|assistant|>\n",
    ),
    (
        "code_task",
        "<|system|>\nYou are an expert Python programmer.\n<|user|>\nWrite a function to reverse a string.\n<|assistant|>\n",
    ),
    ("code_syntax", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"),
]

# Expanded bank for PCA / cross-model similarity (many prompts × many tokens).
EXPANDED_PROBES: list[tuple[str, str]] = CORE_PROBES + [
    ("neutral_fact", "Water boils at 100 degrees Celsius at sea level. Ice melts at"),
    ("neutral_fact", "The largest planet in our solar system is"),
    ("neutral_fact", "Photosynthesis converts sunlight into"),
    ("chat_qa", "<|system|>\nYou are helpful.\n<|user|>\nExplain gravity in one sentence.\n<|assistant|>\n"),
    ("chat_qa", "<|system|>\nYou are helpful.\n<|user|>\nList three colors.\n<|assistant|>\n"),
    ("chat_qa", "<|system|>\nYou are helpful.\n<|user|>\nWhat is the capital of Japan?\n<|assistant|>\n"),
    ("chat_qa", "<|system|>\nYou are helpful.\n<|user|>\nSummarize why the sky is blue.\n<|assistant|>\n"),
    (
        "code_task",
        "<|system|>\nYou are a Python expert.\n<|user|>\nWrite a function to check if a string is a palindrome.\n<|assistant|>\n",
    ),
    (
        "code_task",
        "<|system|>\nYou are a Python expert.\n<|user|>\nWrite a function to compute factorial.\n<|assistant|>\n",
    ),
    (
        "code_task",
        "<|system|>\nYou are a Python expert.\n<|user|>\nWrite a function to merge two sorted lists.\n<|assistant|>\n",
    ),
    (
        "code_task",
        "<|system|>\nYou are a Python expert.\n<|user|>\nWrite a function to count word frequencies in a text.\n<|assistant|>\n",
    ),
    ("code_syntax", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if"),
    ("code_syntax", "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return"),
    ("code_syntax", "import json\n\ndef load_config(path):\n    with open(path) as f:\n        return"),
    ("code_syntax", "async def fetch(url):\n    response = await client.get(url)\n    return"),
    ("embodied", "<|system|>\nYou are a household robot.\n<|user|>\nPick up the red mug on the table.\n<|assistant|>\n"),
    ("embodied", "<|system|>\nYou are a household robot.\n<|user|>\nOpen the drawer and place the spoon inside.\n<|assistant|>\n"),
    ("spatial", "The book is on the table. The lamp is to the left of the book. The phone is behind the lamp. The phone is"),
    ("spatial", "Mary moved to the kitchen. John went to the hallway. Mary is in the"),
]

PROBE_COLORS = {
    "neutral_fact": "#64748b",
    "chat_qa": "#2563eb",
    "code_task": "#dc2626",
    "code_syntax": "#f97316",
    "embodied": "#16a34a",
    "spatial": "#9333ea",
}
MODEL_MARKERS = {"chat": "o", "code": "s", "midtrain": "^"}
MODEL_COLORS = {"chat": "#2563eb", "code": "#dc2626", "midtrain": "#16a34a"}
PROBE_MARKERS = {
    "neutral_fact": "o",
    "chat_qa": "s",
    "code_task": "^",
    "code_syntax": "D",
    "embodied": "v",
    "spatial": "P",
}


@dataclass
class ModelSpec:
    label: str
    checkpoint: Path
    config: Path


@dataclass
class ProbeCapture:
    label: str
    probe: str
    prompt: str
    hidden_states: list[torch.Tensor]
    exit_lambdas: list[float]
    exit_probs: list[float]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_probe_bank(path: Path | None, *, expanded: bool) -> list[tuple[str, str]]:
    if path is not None:
        data = load_yaml(path)
        rows = data.get("probes") or data.get("sources") or data
        if not isinstance(rows, list):
            raise ValueError(f"expected list of probes in {path}")
        out: list[tuple[str, str]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append((str(row["name"]), str(row["prompt"])))
            elif isinstance(row, (list, tuple)) and len(row) == 2:
                out.append((str(row[0]), str(row[1])))
        return out
    return EXPANDED_PROBES if expanded else CORE_PROBES


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_model_arg(raw: str) -> ModelSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"expected label:checkpoint:config, got {raw!r}")
    label, checkpoint, config = parts
    return ModelSpec(label=label, checkpoint=Path(checkpoint), config=Path(config))


@torch.no_grad()
def capture_probe(
    *,
    model: RexForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    probe_name: str,
    prompt: str,
) -> ProbeCapture:
    model.eval()
    saved_threshold = model.cfg.early_exit_threshold
    model.cfg.early_exit_threshold = None
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    try:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            hidden_states, _, _ = model._recurrence_forward(
                input_ids,
                normalize=True,
                num_recurrence_steps=None,
                early_exit_threshold=None,
            )
    finally:
        model.cfg.early_exit_threshold = saved_threshold

    lambdas: list[float] = []
    if model.exit_gate is not None:
        for hidden in hidden_states:
            token_lambda = torch.sigmoid(model.exit_gate(hidden).squeeze(-1)).mean().item()
            lambdas.append(float(token_lambda))
        exit_probs = exit_distribution(torch.tensor(lambdas, device=device).unsqueeze(0)).squeeze(0).tolist()
    else:
        exit_probs = []

    return ProbeCapture(
        label="",
        probe=probe_name,
        prompt=prompt,
        hidden_states=[h.detach().float().cpu() for h in hidden_states],
        exit_lambdas=lambdas,
        exit_probs=[float(p) for p in exit_probs],
    )


def _pool_tokens(hidden: torch.Tensor, last_n: int = 8) -> np.ndarray:
    tail = hidden[0, -last_n:, :]
    return tail.mean(dim=0).numpy()


def _sample_token_vectors(
    hidden: torch.Tensor,
    *,
    max_tokens: int,
    seed: int,
) -> np.ndarray:
    """Return up to max_tokens rows [n, dim] from a [1, seq, dim] tensor."""
    seq_len = hidden.size(1)
    if seq_len <= max_tokens:
        idx = torch.arange(seq_len)
    else:
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(seq_len), max_tokens))
    return hidden[0, idx, :].numpy()


def _recurrence_cosines(capture: ProbeCapture, *, last_n: int) -> list[float]:
    if len(capture.hidden_states) < 2:
        return [1.0]
    final = _pool_tokens(capture.hidden_states[-1], last_n)
    out: list[float] = []
    for hidden in capture.hidden_states:
        vec = _pool_tokens(hidden, last_n)
        num = float(np.dot(vec, final))
        den = float(np.linalg.norm(vec) * np.linalg.norm(final) + 1e-8)
        out.append(num / den)
    return out


def _centered_kernel(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    return x @ x.T


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    kx = _centered_kernel(x)
    ky = _centered_kernel(y)
    hsic = float(np.sum(kx * ky))
    norm = float(np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)) + 1e-8)
    return hsic / norm


def plot_recurrence_trajectories(captures: list[ProbeCapture], out_path: Path, *, last_n: int) -> None:
    fig, axes = plt.subplots(1, len({c.probe for c in captures}), figsize=(4 * len({c.probe for c in captures}), 4), squeeze=False)
    probes = sorted({c.probe for c in captures})
    for ax, probe in zip(axes[0], probes):
        for cap in captures:
            if cap.probe != probe:
                continue
            steps = np.arange(1, len(cap.hidden_states) + 1)
            cosines = _recurrence_cosines(cap, last_n=last_n)
            ax.plot(steps, cosines, marker="o", label=cap.label)
        ax.set_title(probe.replace("_", " "))
        ax.set_xlabel("Recurrence step")
        ax.set_ylabel("Cosine sim to final step")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Recurrence convergence (LoopLM hidden-state trajectory)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_exit_gates(captures: list[ProbeCapture], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.35
    offsets = np.linspace(-0.2, 0.2, max(1, len({c.label for c in captures})))
    for offset, label in zip(offsets, sorted({c.label for c in captures})):
        for cap in [c for c in captures if c.label == label]:
            if not cap.exit_lambdas:
                continue
            x = np.arange(len(cap.exit_lambdas))
            ax.bar(x + offset, cap.exit_lambdas, width=width, alpha=0.7, label=f"{cap.label}/{cap.probe}")
    ax.set_xlabel("Recurrence step")
    ax.set_ylabel("Mean exit gate λ")
    ax.set_title("Exit gate profile by model and probe (core probes)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@dataclass
class TokenMatrix:
    matrix: np.ndarray
    model_labels: list[str]
    probe_labels: list[str]
    prompt_keys: list[str]


def collect_token_matrix(
    captures: list[ProbeCapture],
    *,
    max_tokens_per_prompt: int,
    recurrence_step: int,
    seed: int,
) -> TokenMatrix:
    rows: list[np.ndarray] = []
    model_labels: list[str] = []
    probe_labels: list[str] = []
    prompt_keys: list[str] = []
    for cap in captures:
        step_idx = recurrence_step if recurrence_step >= 0 else len(cap.hidden_states) + recurrence_step
        hidden = cap.hidden_states[step_idx]
        vecs = _sample_token_vectors(hidden, max_tokens=max_tokens_per_prompt, seed=seed + hash(cap.prompt) % 9973)
        rows.append(vecs)
        model_labels.extend([cap.label] * vecs.shape[0])
        probe_labels.extend([cap.probe] * vecs.shape[0])
        prompt_keys.extend([f"{cap.probe}::{cap.prompt}"] * vecs.shape[0])
    return TokenMatrix(
        matrix=np.concatenate(rows, axis=0),
        model_labels=model_labels,
        probe_labels=probe_labels,
        prompt_keys=prompt_keys,
    )


def _step_title(recurrence_step: int) -> str:
    return str(recurrence_step) if recurrence_step >= 0 else f"R{recurrence_step}"


def plot_pca_by_model_type(
    token_data: TokenMatrix,
    out_path: Path,
    *,
    recurrence_step: int,
    max_tokens_per_prompt: int,
    seed: int,
) -> None:
    coords = PCA(n_components=2, random_state=seed).fit_transform(token_data.matrix)
    models = sorted(set(token_data.model_labels))
    probes = sorted(set(token_data.probe_labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    for model in models:
        for probe in probes:
            mask = [
                (m == model and p == probe)
                for m, p in zip(token_data.model_labels, token_data.probe_labels)
            ]
            if not any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=22,
                alpha=0.5,
                c=MODEL_COLORS.get(model, "#64748b"),
                marker=PROBE_MARKERS.get(probe, "o"),
                edgecolors="white",
                linewidths=0.25,
                label=f"{model} / {probe.replace('_', ' ')}",
            )
    ax.set_title(
        f"PCA colored by model type (step {_step_title(recurrence_step)}, "
        f"{len(token_data.matrix)} token points, up to {max_tokens_per_prompt}/prompt)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca_by_probe(
    token_data: TokenMatrix,
    out_path: Path,
    *,
    recurrence_step: int,
    max_tokens_per_prompt: int,
    seed: int,
) -> None:
    coords = PCA(n_components=2, random_state=seed).fit_transform(token_data.matrix)
    models = sorted(set(token_data.model_labels))
    probes = sorted(set(token_data.probe_labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    for model in models:
        for probe in probes:
            mask = [
                (m == model and p == probe)
                for m, p in zip(token_data.model_labels, token_data.probe_labels)
            ]
            if not any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.45,
                marker=MODEL_MARKERS.get(model, "o"),
                c=PROBE_COLORS.get(probe, "#94a3b8"),
                edgecolors="none",
                label=f"{model}/{probe}",
            )
    ax.set_title(
        f"PCA colored by probe category (step {_step_title(recurrence_step)}, "
        f"{len(token_data.matrix)} token points)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lda_model_separation(
    token_data: TokenMatrix,
    out_path: Path,
    *,
    recurrence_step: int,
    seed: int,
) -> dict[str, float]:
    models = sorted(set(token_data.model_labels))
    if len(models) < 2:
        return {}

    y = np.array([models.index(label) for label in token_data.model_labels])
    n_pca = min(50, token_data.matrix.shape[0] - 1, token_data.matrix.shape[1])
    reduced = PCA(n_components=n_pca, random_state=seed).fit_transform(token_data.matrix)
    lda = LinearDiscriminantAnalysis(n_components=min(len(models) - 1, 1))
    scores = lda.fit_transform(reduced, y).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    for idx, model in enumerate(models):
        mask = np.array(token_data.model_labels) == model
        axes[0].hist(
            scores[mask],
            bins=40,
            alpha=0.55,
            color=MODEL_COLORS.get(model, "#64748b"),
            label=model,
            density=True,
        )
    axes[0].set_title(f"LDA axis separating model type (step {_step_title(recurrence_step)})")
    axes[0].set_xlabel("LDA score")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    chat_mask = np.array(token_data.model_labels) == models[0]
    code_mask = np.array(token_data.model_labels) == models[1]
    chat_mean = float(scores[chat_mask].mean())
    code_mean = float(scores[code_mask].mean())
    pooled_std = float(np.std(scores) + 1e-8)
    separation = abs(chat_mean - code_mean) / pooled_std

    axes[1].bar(["mean gap / σ"], [separation], color="#475569")
    axes[1].set_ylim(0, max(1.5, separation * 1.2))
    axes[1].set_title("Model-type separation")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"lda_separation_sigma": separation}


def plot_kmeans_by_model_type(
    token_data: TokenMatrix,
    out_path: Path,
    *,
    recurrence_step: int,
    seed: int,
) -> dict[str, float]:
    models = sorted(set(token_data.model_labels))
    if len(models) < 2:
        return {}

    y_true = np.array([models.index(label) for label in token_data.model_labels])
    n_pca = min(50, token_data.matrix.shape[0] - 1, token_data.matrix.shape[1])
    reduced = PCA(n_components=n_pca, random_state=seed).fit_transform(token_data.matrix)
    coords = PCA(n_components=2, random_state=seed).fit_transform(token_data.matrix)

    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(reduced)

    ari = float(adjusted_rand_score(y_true, cluster_ids))
    sil = float(silhouette_score(reduced, cluster_ids))

    cluster_colors = ["#f59e0b", "#06b6d4"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in (0, 1):
        for model in models:
            mask = (cluster_ids == cluster) & (np.array(token_data.model_labels) == model)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                alpha=0.55,
                c=cluster_colors[cluster],
                marker=MODEL_MARKERS.get(model, "o"),
                edgecolors=MODEL_COLORS.get(model, "#111827"),
                linewidths=0.8,
                label=f"cluster {cluster + 1} / {model}",
            )
    ax.set_title(
        f"k=2 clusters on hidden states (step {_step_title(recurrence_step)})\n"
        f"fill = cluster, edge = true model type | ARI={ari:.2f}, silhouette={sil:.2f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"kmeans_ari": ari, "kmeans_silhouette": sil}


def plot_pca_recurrence_pooled(captures: list[ProbeCapture], out_path: Path, *, last_n: int) -> None:
    rows: list[np.ndarray] = []
    meta: list[tuple[str, str, int]] = []
    for cap in captures:
        for step_idx, hidden in enumerate(cap.hidden_states, start=1):
            rows.append(_pool_tokens(hidden, last_n))
            meta.append((cap.label, cap.probe, step_idx))
    matrix = np.stack(rows, axis=0)
    coords = PCA(n_components=2).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for (x, y), (label, probe, step) in zip(coords, meta):
        ax.scatter(
            x,
            y,
            c=PROBE_COLORS.get(probe, "#64748b"),
            marker=MODEL_MARKERS.get(label, markers.get(step, "o")),
            s=55,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title(f"PCA of pooled hidden states ({len(captures)} prompts × recurrence steps)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model_similarity(
    captures: list[ProbeCapture],
    out_path: Path,
    *,
    max_tokens_per_prompt: int,
    seed: int,
    metric: str,
) -> None:
    labels = sorted({c.label for c in captures})
    probes = sorted({c.probe for c in captures})
    if len(labels) < 2:
        return

    matrix = np.zeros((len(probes), len(labels), len(labels)))
    for pi, probe in enumerate(probes):
        per_label: dict[str, list[np.ndarray]] = {label: [] for label in labels}
        for cap in captures:
            if cap.probe != probe:
                continue
            vecs = _sample_token_vectors(
                cap.hidden_states[-1],
                max_tokens=max_tokens_per_prompt,
                seed=seed + hash((cap.label, cap.prompt)) % 9973,
            )
            per_label[cap.label].append(vecs)
        for label in labels:
            if not per_label[label]:
                continue
            per_label[label] = np.concatenate(per_label[label], axis=0)

        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if a not in per_label or b not in per_label:
                    continue
                if i == j:
                    matrix[pi, i, j] = 1.0
                    continue
                xa = per_label[a]
                xb = per_label[b]
                n = min(xa.shape[0], xb.shape[0])
                if n < 4:
                    va = xa.mean(axis=0)
                    vb = xb.mean(axis=0)
                    matrix[pi, i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))
                elif metric == "cka":
                    matrix[pi, i, j] = linear_cka(xa[:n], xb[:n])
                else:
                    va = xa.mean(axis=0)
                    vb = xb.mean(axis=0)
                    matrix[pi, i, j] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8))

    fig, axes = plt.subplots(1, len(probes), figsize=(4.2 * len(probes), 3.8), squeeze=False)
    title_metric = "linear CKA" if metric == "cka" else "mean cosine"
    for ax, probe in zip(axes[0], probes):
        idx = probes.index(probe)
        im = ax.imshow(matrix[idx], vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(probe.replace("_", " "))
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[idx, i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Cross-model similarity ({title_metric}, final step, token samples)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    captures: list[ProbeCapture],
    out_path: Path,
    *,
    last_n: int,
    cluster_metrics: dict[str, float] | None = None,
) -> None:
    lines = ["# Representation analysis summary", ""]
    lines.append(f"- captures: {len(captures)}")
    if cluster_metrics:
        lines.append("")
        lines.append("## Model-type clustering")
        for key, value in cluster_metrics.items():
            lines.append(f"- {key}: {value:.3f}")
    lines.append("")
    for cap in captures:
        cosines = _recurrence_cosines(cap, last_n=last_n)
        lines.append(f"## {cap.label} / {cap.probe}")
        lines.append(f"- recurrence_cosines: {[round(v, 3) for v in cosines]}")
        if cap.exit_lambdas:
            lines.append(f"- exit_lambdas: {[round(v, 3) for v in cap.exit_lambdas]}")
            lines.append(f"- exit_probs: {[round(v, 3) for v in cap.exit_probs]}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as label:checkpoint:config (repeat for comparisons)",
    )
    parser.add_argument("--output-dir", default="benchmark_results/repr-analysis")
    parser.add_argument("--prompts-file", default=None, help="YAML list of {name, prompt} for PCA bank")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--last-n-tokens", type=int, default=8, help="Tokens to mean-pool for trajectory metrics")
    parser.add_argument(
        "--pca-tokens-per-prompt",
        type=int,
        default=24,
        help="Token positions sampled per prompt for dense PCA (default 24)",
    )
    parser.add_argument("--pca-recurrence-step", type=int, default=-1, help="Recurrence step for token PCA (-1 = final)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--similarity-metric",
        choices=["cka", "cosine"],
        default="cka",
        help="Cross-model similarity metric when enough token samples exist",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = [parse_model_arg(raw) for raw in args.model]
    device = resolve_device(args.device)
    dtype_name = args.dtype.lower()
    amp_dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} and device.type == "cuda" else None
    if dtype_name in {"fp16", "float16"} and device.type == "cuda":
        amp_dtype = torch.float16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_probes = CORE_PROBES
    pca_probes = load_probe_bank(Path(args.prompts_file) if args.prompts_file else None, expanded=True)
    all_captures: list[ProbeCapture] = []
    pca_captures: list[ProbeCapture] = []

    for spec in specs:
        cfg = load_yaml(spec.config)
        tokenizer_name = cfg.get("data", {}).get("tokenizer_name", "HuggingFaceTB/SmolLM2-360M")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        model = RexForCausalLM.from_checkpoint(spec.checkpoint, map_location="cpu")
        model.to(device)

        for probe_name, prompt in core_probes:
            cap = capture_probe(
                model=model,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                probe_name=probe_name,
                prompt=prompt,
            )
            cap.label = spec.label
            all_captures.append(cap)

        for probe_name, prompt in pca_probes:
            cap = capture_probe(
                model=model,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                probe_name=probe_name,
                prompt=prompt,
            )
            cap.label = spec.label
            pca_captures.append(cap)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    token_data = collect_token_matrix(
        pca_captures,
        max_tokens_per_prompt=args.pca_tokens_per_prompt,
        recurrence_step=args.pca_recurrence_step,
        seed=args.seed,
    )

    plot_recurrence_trajectories(all_captures, out_dir / "recurrence_trajectories.png", last_n=args.last_n_tokens)
    plot_exit_gates(all_captures, out_dir / "exit_gates.png")
    plot_pca_by_model_type(
        token_data,
        out_dir / "pca_hidden_states.png",
        recurrence_step=args.pca_recurrence_step,
        max_tokens_per_prompt=args.pca_tokens_per_prompt,
        seed=args.seed,
    )
    plot_pca_by_probe(
        token_data,
        out_dir / "pca_by_probe.png",
        recurrence_step=args.pca_recurrence_step,
        max_tokens_per_prompt=args.pca_tokens_per_prompt,
        seed=args.seed,
    )
    cluster_metrics: dict[str, float] = {}
    cluster_metrics.update(
        plot_lda_model_separation(
            token_data,
            out_dir / "lda_model_separation.png",
            recurrence_step=args.pca_recurrence_step,
            seed=args.seed,
        )
    )
    cluster_metrics.update(
        plot_kmeans_by_model_type(
            token_data,
            out_dir / "kmeans_by_model_type.png",
            recurrence_step=args.pca_recurrence_step,
            seed=args.seed,
        )
    )
    plot_pca_recurrence_pooled(
        pca_captures,
        out_dir / "pca_recurrence_pooled.png",
        last_n=args.last_n_tokens,
    )
    plot_cross_model_similarity(
        pca_captures,
        out_dir / "cross_model_similarity.png",
        max_tokens_per_prompt=args.pca_tokens_per_prompt,
        seed=args.seed,
        metric=args.similarity_metric,
    )
    write_summary(all_captures, out_dir / "summary.md", last_n=args.last_n_tokens, cluster_metrics=cluster_metrics)

    n_points = len(token_data.matrix)
    print(f"Wrote analysis to {out_dir}/")
    print(f"  PCA token points: {n_points}")
    print("  pca_hidden_states.png        — PCA colored by chat/code model type")
    print("  lda_model_separation.png     — LDA histogram separating chat vs code")
    print("  kmeans_by_model_type.png     — k=2 clusters vs true model labels")
    print("  pca_by_probe.png             — PCA colored by probe category")
    print("  pca_recurrence_pooled.png    — all prompts × recurrence steps")
    print("  cross_model_similarity.png   — CKA/cosine with token samples")
    print("  recurrence_trajectories.png  — core probes only")
    print("  exit_gates.png               — core probes only")
    if cluster_metrics:
        print(f"  model-type ARI: {cluster_metrics.get('kmeans_ari', float('nan')):.3f}")


if __name__ == "__main__":
    main()
