"""Download and tokenize text data into raw int32 token streams."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
import yaml
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm
from transformers import AutoTokenizer

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "function": "tool",
    "tool": "tool",
    "tools": "tool",
    "system": "system",
}


def _normalize_role(role: str) -> str:
    return _ROLE_ALIASES.get(str(role).strip().lower(), str(role).strip().lower())


def _turn_role(turn: dict[str, Any]) -> str:
    return str(turn.get("role") or turn.get("from") or turn.get("speaker") or "user")


def _turn_content(turn: dict[str, Any]) -> str:
    content = turn.get("content")
    if content is None:
        content = turn.get("value")
    if content is None:
        content = turn.get("text")
    return _stringify_value(content).strip()


def _parse_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _format_conversation(messages: list[dict[str, Any]], system: str | None = None) -> str | None:
    parts: list[str] = []
    if system and system.strip():
        parts.append(f"<|system|>\n{system.strip()}")
    for turn in messages:
        role = _normalize_role(_turn_role(turn))
        content = _turn_content(turn)
        if not content:
            continue
        if role == "system":
            if not parts or not parts[0].startswith("<|system|>"):
                parts.insert(0, f"<|system|>\n{content}")
            continue
        parts.append(f"<|{role}|>\n{content}")
    if not parts:
        return None
    return "\n".join(parts)


def _conversation_row_text(row: dict[str, Any], source_cfg: dict[str, Any], default_system: str | None) -> str | None:
    conversation_column = source_cfg["conversation_column"]
    system_column = source_cfg.get("system_column")
    messages = _parse_messages(row.get(conversation_column))
    system = None
    if system_column:
        system = _stringify_value(row.get(system_column)).strip() or None
    if system is None:
        system = default_system
    return _format_conversation(messages, system)


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


def _format_boolq_row(row: dict[str, Any]) -> str | None:
    passage = _stringify_value(row.get("passage")).strip()
    question = _stringify_value(row.get("question")).strip()
    answer = row.get("answer")
    if isinstance(answer, bool):
        answer_text = "yes" if answer else "no"
    else:
        answer_text = _stringify_value(answer).strip().lower()
    if not passage or not question or not answer_text:
        return None
    return f"Passage:\n{passage}\n\nQuestion: {question}\nAnswer: {answer_text}"


def _format_squad_row(row: dict[str, Any]) -> str | None:
    context = _stringify_value(row.get("context")).strip()
    question = _stringify_value(row.get("question")).strip()
    answers = row.get("answers")
    answer_text = ""
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        if texts:
            answer_text = _stringify_value(texts[0]).strip()
    if not context or not question or not answer_text:
        return None
    return f"Passage:\n{context}\n\nQuestion: {question}\nAnswer: {answer_text}"


def _format_openbookqa_row(row: dict[str, Any]) -> str | None:
    stem = _stringify_value(row.get("question_stem")).strip()
    choices = row.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    key = _stringify_value(row.get("answerKey")).strip()
    answer_text = ""
    for label, text in zip(labels, texts):
        if str(label) == key:
            answer_text = _stringify_value(text).strip()
            break
    if not stem or not answer_text:
        return None
    return f"Question: {stem}\nAnswer: {answer_text}"


def _format_trivia_qa_row(row: dict[str, Any]) -> str | None:
    question = _stringify_value(row.get("question")).strip()
    answer_obj = row.get("answer") or {}
    answer_text = _stringify_value(answer_obj.get("value") or answer_obj.get("normalized_value")).strip()
    context = ""
    entity_pages = row.get("entity_pages") or {}
    contexts = entity_pages.get("wiki_context") or []
    if contexts:
        context = _stringify_value(contexts[0]).strip()
    if not context:
        search_results = row.get("search_results") or {}
        descs = search_results.get("description") or []
        if descs:
            context = _stringify_value(descs[0]).strip()
    if not question or not answer_text:
        return None
    if context:
        return f"Passage:\n{context}\n\nQuestion: {question}\nAnswer: {answer_text}"
    return f"Question: {question}\nAnswer: {answer_text}"


def _format_babi_row(row: dict[str, Any]) -> str | None:
    passage = _stringify_value(row.get("passage")).strip()
    question = _stringify_value(row.get("question")).strip()
    answer = _stringify_value(row.get("answer")).strip()
    if not passage or not question or not answer:
        return None
    return f"Context:\n{passage}\nQuestion: {question}\nAnswer: {answer}"


_ROW_FORMATTERS = {
    "boolq": _format_boolq_row,
    "squad": _format_squad_row,
    "openbookqa": _format_openbookqa_row,
    "trivia_qa": _format_trivia_qa_row,
    "babi": _format_babi_row,
}


def _iter_cauldron_qa(source_cfg: dict[str, Any]) -> Iterable[str]:
    train_split = source_cfg.get("train_split", "train")
    max_docs = source_cfg.get("max_train_docs")
    source_ds = _load_source_dataset(source_cfg, train_split)
    yielded = 0
    for row in source_ds:
        for turn in row.get("texts") or []:
            if not isinstance(turn, dict):
                continue
            user = _stringify_value(turn.get("user")).strip()
            assistant = _stringify_value(turn.get("assistant")).strip()
            if not user or not assistant:
                continue
            yield f"Question: {user}\nAnswer: {assistant}"
            yielded += 1
            if max_docs is not None and yielded >= int(max_docs):
                return


def _iter_cauldron_qa_val(source_cfg: dict[str, Any]) -> Iterable[str]:
    val_split = source_cfg.get("val_split")
    if not val_split:
        return iter(())
    max_docs = source_cfg.get("max_val_docs")
    source_ds = _load_source_dataset(source_cfg, val_split)
    yielded = 0
    for row in source_ds:
        for turn in row.get("texts") or []:
            if not isinstance(turn, dict):
                continue
            user = _stringify_value(turn.get("user")).strip()
            assistant = _stringify_value(turn.get("assistant")).strip()
            if not user or not assistant:
                continue
            yield f"Question: {user}\nAnswer: {assistant}"
            yielded += 1
            if max_docs is not None and yielded >= int(max_docs):
                return


def _row_text_from_source(row: dict[str, Any], source_cfg: dict[str, Any]) -> str | None:
    formatter = source_cfg.get("row_formatter")
    if formatter:
        fn = _ROW_FORMATTERS.get(formatter)
        if fn is None:
            raise ValueError(f"unknown row_formatter: {formatter}")
        return fn(row)
    return _row_text(
        row,
        source_cfg.get("text_column", "text"),
        source_cfg.get("text_template"),
    )


def _row_matches_filter(row: dict[str, Any], row_filter: dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    for key, expected in row_filter.items():
        actual = row.get(key)
        if actual is None and isinstance(metadata, dict):
            actual = metadata.get(key)
        if actual != expected:
            return False
    return True


def _row_matches_min_filter(row: dict[str, Any], row_filter_min: dict[str, float]) -> bool:
    metadata = row.get("metadata")
    for key, minimum in row_filter_min.items():
        actual = row.get(key)
        if actual is None and isinstance(metadata, dict):
            actual = metadata.get(key)
        if actual is None or float(actual) < float(minimum):
            return False
    return True


def _iter_text(
    dataset: Iterable[dict[str, Any]],
    text_column: str | list[str],
    text_template: str | None = None,
    *,
    source_cfg: dict[str, Any] | None = None,
    default_system: str | None = None,
) -> Iterable[str]:
    row_filter = source_cfg.get("row_filter") if source_cfg else None
    row_filter_min = source_cfg.get("row_filter_min") if source_cfg else None
    for row in dataset:
        if row_filter and not _row_matches_filter(row, row_filter):
            continue
        if row_filter_min and not _row_matches_min_filter(row, row_filter_min):
            continue
        if source_cfg and source_cfg.get("conversation_column"):
            text = _conversation_row_text(row, source_cfg, default_system)
        else:
            text = (
                _row_text_from_source(row, source_cfg)
                if source_cfg
                else _row_text(row, text_column, text_template)
            )
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
    parquet_files = source_cfg.get("parquet_files")
    if parquet_files:
        load_kwargs = {"streaming": bool(source_cfg.get("streaming", False))}
        return load_dataset("parquet", data_files=parquet_files, split=split, **load_kwargs)

    dataset_name = source_cfg["dataset_name"]
    dataset_config = source_cfg.get("dataset_config")
    load_kwargs = {"streaming": bool(source_cfg.get("streaming", False))}
    if dataset_config:
        load_kwargs["name"] = dataset_config
    try:
        return load_dataset(dataset_name, split=split, **load_kwargs)
    except DatasetNotFoundError as exc:
        message = str(exc)
        if "gated dataset" in message.lower():
            label = source_cfg.get("name") or dataset_name
            raise DatasetNotFoundError(
                f"{label} ({dataset_name}) is gated on Hugging Face. "
                f"Log in with `huggingface-cli login`, open the dataset page, accept the license, "
                f"then retry. Original error: {message}"
            ) from exc
        raise


def _iter_spartqa_sft(source_cfg: dict[str, Any], *, default_system: str | None = None) -> Iterable[str]:
    split = source_cfg.get("split", "test")
    queries = load_dataset("mteb/SpartQA", "queries", split=split)
    corpus = load_dataset("mteb/SpartQA", "corpus", split=split)
    qrels = load_dataset("mteb/SpartQA", "qrels", split=split)
    qmap = {row["_id"]: row["text"] for row in queries}
    cmap = {row["_id"]: row["text"] for row in corpus}
    system = (default_system or "").strip()
    for row in qrels:
        question = qmap.get(row["query-id"])
        answer = cmap.get(row["corpus-id"])
        if not question or not answer:
            continue
        parts: list[str] = []
        if system:
            parts.append(f"<|system|>\n{system}")
        parts.append(f"<|user|>\n{question.strip()}")
        parts.append(f"<|assistant|>\n{answer.strip()}")
        yield "\n".join(parts)


def _source_texts(source_cfg: dict[str, Any], *, want_val: bool, default_system: str | None = None) -> Iterable[str]:
    if source_cfg.get("cauldron_qa"):
        if want_val:
            return _iter_cauldron_qa_val(source_cfg)
        return _iter_cauldron_qa(source_cfg)

    if source_cfg.get("spartqa_joined"):
        if want_val:
            return iter(())
        return _iter_spartqa_sft(source_cfg, default_system=default_system)

    text_column = source_cfg.get("text_column", "text")
    text_template = source_cfg.get("text_template")
    train_split = source_cfg.get("train_split", "train")
    val_split = source_cfg.get("val_split")
    split_strategy = source_cfg.get("split_strategy", "head")
    max_val_docs = int(source_cfg.get("max_val_docs") or 0)
    iter_kwargs = {"source_cfg": source_cfg, "default_system": default_system}

    if want_val and val_split:
        return _iter_text(_load_source_dataset(source_cfg, val_split), text_column, text_template, **iter_kwargs)
    if val_split:
        return _iter_text(_load_source_dataset(source_cfg, train_split), text_column, text_template, **iter_kwargs)

    source_ds = _load_source_dataset(source_cfg, train_split)
    texts = _iter_text(source_ds, text_column, text_template, **iter_kwargs)
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


def _write_replay_tokens_to_file(
    *,
    replay_bin: Path,
    file: BinaryIO,
    max_tokens: int | None,
    seed: int,
    chunk_min: int = 512,
    chunk_max: int = 4096,
    desc: str,
) -> tuple[int, int]:
    if max_tokens is not None and max_tokens <= 0:
        return 0, 0
    if not replay_bin.exists():
        raise FileNotFoundError(f"replay bin not found: {replay_bin}")

    data = np.memmap(replay_bin, dtype=np.int32, mode="r")
    if data.size <= chunk_min:
        return 0, 0

    rng = random.Random(seed)
    emitted = 0
    docs = 0
    pbar = tqdm(total=max_tokens, desc=desc, unit="tok")
    while max_tokens is None or emitted < max_tokens:
        remaining = None if max_tokens is None else max_tokens - emitted
        upper = min(chunk_max, data.size)
        if remaining is not None:
            upper = min(upper, remaining)
        if upper < chunk_min:
            break
        chunk_len = rng.randint(chunk_min, upper)
        start = rng.randint(0, max(0, data.size - chunk_len))
        chunk = np.asarray(data[start : start + chunk_len], dtype=np.int32)
        chunk.tofile(file)
        emitted += int(chunk.size)
        docs += 1
        pbar.update(int(chunk.size))
    pbar.close()
    return docs, emitted


def _download_mixed_sources(
    *,
    sources: list[dict[str, Any]],
    train_bin: Path,
    val_bin: Path,
    tokenizer: AutoTokenizer,
    default_system: str | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    train_bin.parent.mkdir(parents=True, exist_ok=True)
    val_bin.parent.mkdir(parents=True, exist_ok=True)
    source_meta = []
    train_docs = train_tokens = val_docs = val_tokens = 0
    with open(val_bin, "wb") as val_file, open(train_bin, "wb") as train_file:
        for source_cfg in sources:
            label = source_cfg.get("name") or source_cfg.get("dataset_name") or source_cfg.get("replay_bin", "source")
            if source_cfg.get("replay_bin"):
                replay_train = Path(source_cfg["replay_bin"])
                replay_val = Path(source_cfg.get("replay_val_bin") or source_cfg["replay_bin"])
                replay_seed = int(source_cfg.get("seed", 1337))
                chunk_min = int(source_cfg.get("replay_chunk_min", 512))
                chunk_max = int(source_cfg.get("replay_chunk_max", 4096))
                current_val_docs, current_val_tokens = _write_replay_tokens_to_file(
                    replay_bin=replay_val,
                    file=val_file,
                    max_tokens=source_cfg.get("max_val_tokens"),
                    seed=replay_seed + 1,
                    chunk_min=chunk_min,
                    chunk_max=chunk_max,
                    desc=f"replaying val:{label}",
                )
                current_train_docs, current_train_tokens = _write_replay_tokens_to_file(
                    replay_bin=replay_train,
                    file=train_file,
                    max_tokens=source_cfg.get("max_train_tokens"),
                    seed=replay_seed,
                    chunk_min=chunk_min,
                    chunk_max=chunk_max,
                    desc=f"replaying train:{label}",
                )
                source_meta.append(
                    {
                        "name": label,
                        "replay_bin": str(replay_train),
                        "replay_val_bin": str(replay_val),
                        "max_train_tokens": source_cfg.get("max_train_tokens"),
                        "max_val_tokens": source_cfg.get("max_val_tokens"),
                        "train_docs": current_train_docs,
                        "val_docs": current_val_docs,
                        "train_tokens": current_train_tokens,
                        "val_tokens": current_val_tokens,
                    }
                )
            else:
                name = source_cfg.get("dataset_name") or label
                source_max_val_docs = source_cfg.get("max_val_docs")
                source_max_train_docs = source_cfg.get("max_train_docs")
                current_val_docs, current_val_tokens = _write_tokens_to_file(
                    texts=_source_texts(source_cfg, want_val=True, default_system=default_system),
                    file=val_file,
                    tokenizer=tokenizer,
                    max_docs=source_max_val_docs,
                    desc=f"tokenizing val:{label}",
                )
                current_train_docs, current_train_tokens = _write_tokens_to_file(
                    texts=_source_texts(source_cfg, want_val=False, default_system=default_system),
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
                        "conversation_column": source_cfg.get("conversation_column"),
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
    default_system = dl_cfg.get("default_system")

    if sources:
        source_meta, train_docs, train_tokens, val_docs, val_tokens = _download_mixed_sources(
            sources=sources,
            train_bin=train_bin,
            val_bin=val_bin,
            tokenizer=tokenizer,
            default_system=default_system,
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
