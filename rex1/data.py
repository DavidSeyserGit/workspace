"""Dataset utilities for REX language-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler

IGNORE_LABEL = -100


def mask_path_for(bin_path: str | Path) -> Path:
    path = Path(bin_path)
    return path.with_name(f"{path.stem}.mask.bin")


class MemmapTokenDataset(Dataset):
    """Contiguous next-token chunks backed by a raw int32 token file."""

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        stride: int | None = None,
        max_tokens: int | None = None,
        *,
        assistant_only_loss: bool = False,
        mask_path: str | Path | None = None,
    ):
        self.path = Path(path)
        self.block_size = int(block_size)
        self.stride = int(stride or 1)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.assistant_only_loss = bool(assistant_only_loss)
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not self.path.exists():
            raise FileNotFoundError(f"token file not found: {self.path}")
        if self.path.stat().st_size % np.dtype(np.int32).itemsize != 0:
            raise ValueError(f"{self.path} is not a raw int32 token file")
        self.tokens = np.memmap(self.path, dtype=np.int32, mode="r")
        self.num_tokens = len(self.tokens) if self.max_tokens is None else min(len(self.tokens), self.max_tokens)
        if self.num_tokens <= self.block_size:
            raise ValueError(f"{self.path} has {self.num_tokens} usable tokens; need > block_size")

        resolved_mask = Path(mask_path) if mask_path is not None else mask_path_for(self.path)
        if self.assistant_only_loss and resolved_mask.exists():
            self.mask = np.memmap(resolved_mask, dtype=np.uint8, mode="r")
            if self.mask.size != len(self.tokens):
                raise ValueError(
                    f"mask file {resolved_mask} length {self.mask.size} != token file length {len(self.tokens)}"
                )
        else:
            self.mask = None
            if self.assistant_only_loss and not resolved_mask.exists():
                print(
                    f"warning: assistant_only_loss enabled but no mask at {resolved_mask}; "
                    f"using full-sequence loss for {self.path}"
                )

    def __len__(self) -> int:
        return ((self.num_tokens - self.block_size) // self.stride) + 1

    def _labels_from_chunk(self, input_ids: torch.Tensor, supervise: np.ndarray | None) -> torch.Tensor:
        labels = input_ids.clone()
        if supervise is None:
            return labels
        labels[0] = IGNORE_LABEL
        for idx in range(1, input_ids.size(0)):
            if supervise[idx] == 0:
                labels[idx] = IGNORE_LABEL
        return labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        chunk = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
        input_ids = torch.from_numpy(chunk.copy())
        supervise = None
        if self.mask is not None:
            supervise = np.asarray(self.mask[start : start + self.block_size], dtype=np.uint8)
        labels = self._labels_from_chunk(input_ids, supervise)
        return {"input_ids": input_ids, "labels": labels}


def _build_token_dataset(
    spec: str | Path | dict[str, Any],
    *,
    block_size: int,
    stride: int,
    assistant_only_loss: bool,
) -> MemmapTokenDataset:
    if isinstance(spec, dict):
        path = spec["path"]
        max_tokens = spec.get("max_tokens")
        spec_assistant_only = spec.get("assistant_only_loss", assistant_only_loss)
        mask_path = spec.get("mask_path")
    else:
        path = spec
        max_tokens = None
        spec_assistant_only = assistant_only_loss
        mask_path = None
    return MemmapTokenDataset(
        path,
        block_size,
        stride=stride,
        max_tokens=max_tokens,
        assistant_only_loss=spec_assistant_only,
        mask_path=mask_path,
    )


def build_token_dataset(
    specs: str | Path | dict[str, Any] | list[str | Path | dict[str, Any]],
    *,
    block_size: int,
    stride: int,
    assistant_only_loss: bool = False,
) -> Dataset:
    if isinstance(specs, list):
        datasets = [
            _build_token_dataset(spec, block_size=block_size, stride=stride, assistant_only_loss=assistant_only_loss)
            for spec in specs
        ]
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)
    return _build_token_dataset(specs, block_size=block_size, stride=stride, assistant_only_loss=assistant_only_loss)


def collate_token_blocks(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_dataloaders(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    *,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    block_size = int(data_cfg.get("block_size", 2048))
    stride = int(data_cfg.get("stride", 1))
    if data_cfg.get("pack_sequences", False):
        stride = block_size
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 2))
    assistant_only_loss = bool(data_cfg.get("assistant_only_loss", False))
    train_specs = data_cfg.get("train_bins") or data_cfg["train_bin"]
    val_specs = data_cfg.get("val_bins") or data_cfg.get("val_bin")

    train_ds = build_token_dataset(
        train_specs,
        block_size=block_size,
        stride=stride,
        assistant_only_loss=assistant_only_loss,
    )
    val_ds = (
        build_token_dataset(val_specs, block_size=block_size, stride=stride, assistant_only_loss=assistant_only_loss)
        if val_specs
        else None
    )
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_token_blocks,
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_token_blocks,
        )
    return train_loader, val_loader, train_sampler
