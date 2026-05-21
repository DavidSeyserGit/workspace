"""Download and tokenize text data into raw int32 token streams."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def _load_yaml(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role and content:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(" ".join(_stringify_value(v) for v in item.values()))
            else:
                parts.append(_stringify_value(item))
        return "\n".join(part for part in parts if part.strip())
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_stringify_value(val)}" for key, val in value.items())
    return str(value)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def _row_text(row: dict[str, Any], text_column: str | list[str], text_template: str | None = None) -> str | None:
    if text_template:
        values = _SafeFormatDict({key: _stringify_value(value).strip() for key, value in row.items()})
        text = text_template.format_map(values)
    elif isinstance(text_column, list):
        parts = [_stringify_value(row.get(column)).strip() for column in text_column]
        text = "\n".join(part for part in parts if part)
    else:
        value = row.get(text_column)
        text = _stringify_value(value)
    if isinstance(text, str) and text.strip():
        return text
    return None


def _iter_text(dataset: Iterable[dict[str, Any]], text_column: str | list[str], text_template: str | None = None) -> Iterable[str]:
    for row in dataset:
        text = _row_text(row, text_column, text_template)
        if text is not None:
            yield text


def _skip(items: Iterable[str], n: int) -> Iterable[str]:
    iterator = iter(items)
    for _ in range(max(0, n)):
        try:
            next(iterator)
        except StopIteration:
            return
    yield from iterator


def _write_tokens_to_file(
    *,
    texts: Iterable[str],
    file: BinaryIO,
    tokenizer: AutoTokenizer,
    max_docs: int | None,
    desc: str,
) -> tuple[int, int]:
    eos_id = tokenizer.eos_token_id
    docs = 0
    tokens = 0
    for text in tqdm(texts, desc=desc):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            ids.append(eos_id)
        if not ids:
            continue
        np.asarray(ids, dtype=np.int32).tofile(file)
        docs += 1
        tokens += len(ids)
        if max_docs is not None and docs >= max_docs:
            break
    return docs, tokens


def _write_tokens(
    *,
    texts: Iterable[str],
    out_path: Path,
    tokenizer: AutoTokenizer,
    max_docs: int | None,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        return _write_tokens_to_file(
            texts=texts,
            file=f,
            tokenizer=tokenizer,
            max_docs=max_docs,
            desc=f"tokenizing {out_path.name}",
        )


def _load_source_dataset(source_cfg: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    dataset_name = source_cfg["dataset_name"]
    dataset_config = source_cfg.get("dataset_config")
    load_kwargs = {"streaming": bool(source_cfg.get("streaming", False))}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    return load_dataset(dataset_name, split=split, **load_kwargs)


def _source_texts(source_cfg: dict[str, Any], *, want_val: bool) -> Iterable[str]:
    text_column = source_cfg.get("text_column", "text")
    text_template = source_cfg.get("text_template")
    train_split = source_cfg.get("train_split", "train")
    val_split = source_cfg.get("val_split")
    split_strategy = source_cfg.get("split_strategy", "head")
    max_val_docs = int(source_cfg.get("max_val_docs") or 0)

    if want_val and val_split:
        return _iter_text(_load_source_dataset(source_cfg, val_split), text_column, text_template)
    if val_split:
        return _iter_text(_load_source_dataset(source_cfg, train_split), text_column, text_template)

    source_ds = _load_source_dataset(source_cfg, train_split)
    texts = _iter_text(source_ds, text_column, text_template)
    if split_strategy == "head":
        return texts if want_val else _skip(texts, max_val_docs)
    if split_strategy != "random":
        raise ValueError(f"unsupported split_strategy for mixed source: {split_strategy}")

    rng = random.Random(int(source_cfg.get("seed", 1337)))
    val_fraction = float(source_cfg.get("val_fraction", 0.005))

    def split_texts() -> Iterable[str]:
        yielded = 0
        limit = source_cfg.get("max_val_docs") if want_val else source_cfg.get("max_train_docs")
        for text in texts:
            is_val = rng.random() < val_fraction
            if is_val == want_val:
                yielded += 1
                yield text
            if limit is not None and yielded >= int(limit):
                break

    return split_texts()


def _download_mixed_sources(
    *,
    sources: list[dict[str, Any]],
    train_bin: Path,
    val_bin: Path,
    tokenizer: AutoTokenizer,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    train_bin.parent.mkdir(parents=True, exist_ok=True)
    val_bin.parent.mkdir(parents=True, exist_ok=True)
    source_meta = []
    train_docs = train_tokens = val_docs = val_tokens = 0
    with open(val_bin, "wb") as val_file, open(train_bin, "wb") as train_file:
        for source_cfg in sources:
            name = source_cfg["dataset_name"]
            label = source_cfg.get("name") or name
            source_max_val_docs = source_cfg.get("max_val_docs")
            source_max_train_docs = source_cfg.get("max_train_docs")
            current_val_docs, current_val_tokens = _write_tokens_to_file(
                texts=_source_texts(source_cfg, want_val=True),
                file=val_file,
                tokenizer=tokenizer,
                max_docs=source_max_val_docs,
                desc=f"tokenizing val:{label}",
            )
            current_train_docs, current_train_tokens = _write_tokens_to_file(
                texts=_source_texts(source_cfg, want_val=False),
                file=train_file,
                tokenizer=tokenizer,
                max_docs=source_max_train_docs,
                desc=f"tokenizing train:{label}",
            )
            source_meta.append(
                {
                    "name": label,
                    "dataset_name": name,
                    "dataset_config": source_cfg.get("dataset_config"),
                    "text_column": source_cfg.get("text_column", "text"),
                    "text_template": source_cfg.get("text_template"),
                    "train_docs": current_train_docs,
                    "val_docs": current_val_docs,
                    "train_tokens": current_train_tokens,
                    "val_tokens": current_val_tokens,
                }
            )
            train_docs += current_train_docs
            train_tokens += current_train_tokens
            val_docs += current_val_docs
            val_tokens += current_val_tokens
    return source_meta, train_docs, train_tokens, val_docs, val_tokens


def download_and_tokenize(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    dl_cfg = data_cfg.get("download", {})
    tokenizer_name = data_cfg.get("tokenizer_name", "gpt2")
    dataset_name = dl_cfg.get("dataset_name", "roneneldan/TinyStories")
    dataset_config = dl_cfg.get("dataset_config")
    text_column = dl_cfg.get("text_column", "text")
    train_split = dl_cfg.get("train_split", "train")
    val_split = dl_cfg.get("val_split")
    val_fraction = float(dl_cfg.get("val_fraction", 0.005))
    split_strategy = dl_cfg.get("split_strategy", "random")
    seed = int(dl_cfg.get("seed", 1337))
    max_train_docs = dl_cfg.get("max_train_docs")
    max_val_docs = dl_cfg.get("max_val_docs")
    streaming = bool(dl_cfg.get("streaming", False))

    train_bin = Path(data_cfg.get("train_bin", "data/train.bin"))
    val_bin = Path(data_cfg.get("val_bin", "data/val.bin"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    sources = dl_cfg.get("sources")

    if sources:
        source_meta, train_docs, train_tokens, val_docs, val_tokens = _download_mixed_sources(
            sources=sources,
            train_bin=train_bin,
            val_bin=val_bin,
            tokenizer=tokenizer,
        )
        meta = {
            "dataset_name": "mixed",
            "tokenizer_name": tokenizer_name,
            "train_bin": str(train_bin),
            "val_bin": str(val_bin),
            "train_docs": train_docs,
            "val_docs": val_docs,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "dtype": "int32",
            "sources": source_meta,
        }
        meta_path = train_bin.parent / "dataset_meta.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        return meta

    load_kwargs = {"streaming": streaming}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    if val_split:
        train_ds = load_dataset(dataset_name, split=train_split, **load_kwargs)
        val_ds = load_dataset(dataset_name, split=val_split, **load_kwargs)
        train_docs, train_tokens = _write_tokens(
            texts=_iter_text(train_ds, text_column),
            out_path=train_bin,
            tokenizer=tokenizer,
            max_docs=max_train_docs,
        )
        val_docs, val_tokens = _write_tokens(
            texts=_iter_text(val_ds, text_column),
            out_path=val_bin,
            tokenizer=tokenizer,
            max_docs=max_val_docs,
        )
    elif split_strategy == "head":
        if max_val_docs is None:
            raise ValueError("split_strategy=head requires max_val_docs")
        source_ds = load_dataset(dataset_name, split=train_split, **load_kwargs)
        val_docs, val_tokens = _write_tokens(
            texts=_iter_text(source_ds, text_column),
            out_path=val_bin,
            tokenizer=tokenizer,
            max_docs=max_val_docs,
        )
        source_ds = load_dataset(dataset_name, split=train_split, **load_kwargs)
        train_docs, train_tokens = _write_tokens(
            texts=_skip(_iter_text(source_ds, text_column), max_val_docs),
            out_path=train_bin,
            tokenizer=tokenizer,
            max_docs=max_train_docs,
        )
    else:
        rng = random.Random(seed)
        source_ds = load_dataset(dataset_name, split=train_split, **load_kwargs)

        def split_texts(want_val: bool) -> Iterable[str]:
            seen = 0
            yielded = 0
            for text in _iter_text(source_ds, text_column):
                seen += 1
                is_val = rng.random() < val_fraction
                if is_val == want_val:
                    yielded += 1
                    yield text
                limit = max_val_docs if want_val else max_train_docs
                if limit is not None and yielded >= limit:
                    break
                if not streaming and max_train_docs is not None and seen > max_train_docs * 4:
                    break

        val_docs, val_tokens = _write_tokens(
            texts=split_texts(True),
            out_path=val_bin,
            tokenizer=tokenizer,
            max_docs=max_val_docs,
        )
        source_ds = load_dataset(dataset_name, split=train_split, **load_kwargs)
        rng = random.Random(seed)
        train_docs, train_tokens = _write_tokens(
            texts=split_texts(False),
            out_path=train_bin,
            tokenizer=tokenizer,
            max_docs=max_train_docs,
        )

    meta = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "tokenizer_name": tokenizer_name,
        "text_column": text_column,
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dtype": "int32",
    }
    meta_path = train_bin.parent / "dataset_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config-mixed-2-v3.yaml", help="Path to training YAML config")
    parser.add_argument("--dataset-name", default=None, help="Override Hugging Face dataset name")
    parser.add_argument("--max-train-docs", type=int, default=None, help="Override train doc cap")
    parser.add_argument("--max-val-docs", type=int, default=None, help="Override validation doc cap")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = _load_yaml(args.config)
    cfg.setdefault("data", {}).setdefault("download", {})
    if args.dataset_name is not None:
        cfg["data"]["download"]["dataset_name"] = args.dataset_name
    if args.max_train_docs is not None:
        cfg["data"]["download"]["max_train_docs"] = args.max_train_docs
    if args.max_val_docs is not None:
        cfg["data"]["download"]["max_val_docs"] = args.max_val_docs
    meta = download_and_tokenize(cfg)
    print(yaml.safe_dump(meta, sort_keys=False))


if __name__ == "__main__":
    main()
