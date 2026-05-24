"""Train REX from a YAML config."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from data import build_dataloaders
from model import RexConfig, RexForCausalLM

try:
    from benchmark import benchmark_wandb_payload, parse_benchmark_tasks, run_model_benchmarks, save_benchmark_snapshot
except ImportError:  # pragma: no cover - optional during minimal installs
    benchmark_wandb_payload = None
    parse_benchmark_tasks = None
    run_model_benchmarks = None
    save_benchmark_snapshot = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_distributed() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, True


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if hasattr(model, "module") else model
    return getattr(module, "_orig_mod", module)


def set_seed(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def cosine_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def warmup_stable_cosine_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    stable_steps: int,
    max_steps: int,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step < stable_steps:
        return base_lr
    decay_start = stable_steps
    progress = (step - decay_start) / max(1, max_steps - decay_start)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def resolve_lr(step: int, train_cfg: dict[str, Any], max_steps: int) -> float:
    base_lr = float(train_cfg["learning_rate"])
    min_lr = float(train_cfg.get("min_lr", 0.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 1000))
    schedule = str(train_cfg.get("lr_schedule", "cosine"))
    if schedule == "warmup_stable_cosine":
        stable_steps = int(train_cfg.get("stable_steps", max_steps // 2))
        return warmup_stable_cosine_lr(
            step,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            max_steps=max_steps,
        )
    return cosine_lr(step, base_lr=base_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps)


def configure_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    decay = []
    no_decay = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        eps=float(train_cfg.get("eps", 1e-8)),
    )


def init_wandb(cfg: dict[str, Any], model: RexForCausalLM, train_cfg: dict[str, Any]) -> Any | None:
    wandb_cfg = train_cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging is enabled, but wandb is not installed") from exc

    run = wandb.init(
        project=wandb_cfg.get("project", "rex"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode", "online"),
        config=cfg,
    )
    wandb.log({"model/parameters": model.parameter_count()}, step=0)
    if wandb_cfg.get("watch", False):
        wandb.watch(model, log=wandb_cfg.get("watch_log", "gradients"), log_freq=int(wandb_cfg.get("watch_log_freq", 100)))
    return run


@torch.no_grad()
def run_validation(
    *,
    eval_mode: str,
    eval_cfg: dict[str, Any],
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader | None,
    cfg: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    eval_batches: int,
    step: int,
    out_dir: Path,
    wandb_run: Any | None,
) -> None:
    if eval_mode == "benchmarks":
        if run_model_benchmarks is None:
            raise ImportError("benchmark mode requires benchmark.py")
        bench_cfg = eval_cfg.get("benchmarks", {})
        tasks = parse_benchmark_tasks(
            bench_cfg.get("tasks", "hellaswag,arc_easy,sciq,openbookqa,val_ppl")
        )
        val_batches = int(bench_cfg.get("val_batches", eval_batches))
        batch_size = int(bench_cfg.get("batch_size", 2))
        mc_examples = int(bench_cfg.get("mc_examples", 50))
        recurrence_steps = bench_cfg.get("recurrence_steps")
        recurrence_steps = int(recurrence_steps) if recurrence_steps is not None else None
        early_exit_threshold: float | None | str = (
            bench_cfg["early_exit_threshold"] if "early_exit_threshold" in bench_cfg else "inherit"
        )
        quiet = bool(bench_cfg.get("quiet", True))
        print(f"\nstep {step}: running benchmarks ({', '.join(sorted(tasks))})")
        results = run_model_benchmarks(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            tasks=tasks,
            val_batches=val_batches,
            batch_size=batch_size,
            mc_examples=mc_examples,
            recurrence_steps=recurrence_steps,
            early_exit_threshold=early_exit_threshold,
            checkpoint_label=f"step{step}",
            quiet=quiet,
        )
        if bool(bench_cfg.get("save", True)):
            json_path = save_benchmark_snapshot(results, out_dir, step)
            print(f"benchmark/json: {json_path}")
        if wandb_run is not None:
            wandb_run.log(benchmark_wandb_payload(results), step=step)
        return

    val_loss = evaluate(model, val_loader, device, amp_dtype, eval_batches)
    if val_loss is not None:
        val_ppl = math.exp(min(20, val_loss))
        print(f"\nstep {step}: val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
        if wandb_run is not None:
            wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)


@torch.no_grad()
def evaluate(
    model: RexForCausalLM,
    val_loader: torch.utils.data.DataLoader | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int,
) -> float | None:
    if val_loader is None:
        return None
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids, labels=labels)
        losses.append(float(out["loss"].item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_to_save.cfg.to_dict(),
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    *,
    path: str | None,
    model: RexForCausalLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def load_init_weights(
    *,
    path: str,
    model: RexForCausalLM,
    device: torch.device,
    rank: int,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    prior_step = int(checkpoint.get("step", 0))
    if is_main_process(rank):
        print(f"init-from {path}: loaded weights from step {prior_step}, starting at step 0")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config-mixed-2-v3.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Resume model, optimizer, and step counter")
    parser.add_argument(
        "--init-from",
        default=None,
        help="Load model weights only (fresh optimizer, step 0). Use when starting a new stage.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rank, world_size, local_rank, distributed = setup_distributed()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = dict(cfg.get("model", {}))

    set_seed(int(train_cfg.get("seed", 1337)), rank=rank)
    requested_device = str(train_cfg.get("device", "auto"))
    if requested_device == "auto":
        if distributed:
            requested_device = f"cuda:{local_rank}"
        else:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    dtype_name = str(train_cfg.get("dtype", "bfloat16")).lower()
    amp_dtype = None
    if device.type == "cuda":
        if dtype_name in {"bf16", "bfloat16"}:
            amp_dtype = torch.bfloat16
        elif dtype_name in {"fp16", "float16"}:
            amp_dtype = torch.float16

    tokenizer_name = data_cfg.get("tokenizer_name", "gpt2")
    if is_main_process(rank):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    else:
        tokenizer = None
    if distributed:
        objects = [tokenizer]
        dist.broadcast_object_list(objects, src=0)
        tokenizer = objects[0]
    model_cfg["vocab_size"] = int(model_cfg.get("vocab_size") or len(tokenizer))
    model_cfg["max_seq_len"] = int(model_cfg.get("max_seq_len") or data_cfg.get("block_size", 2048))
    rex_cfg = RexConfig.from_dict(model_cfg)

    train_loader, val_loader, train_sampler = build_dataloaders(data_cfg, train_cfg, distributed=distributed)
    model = RexForCausalLM(rex_cfg).to(device)
    param_count = model.parameter_count()
    if is_main_process(rank):
        print(f"REX parameters: {param_count / 1e6:.1f}M")
    optimizer = configure_optimizer(model, train_cfg)
    resume_path = args.resume or train_cfg.get("resume")
    init_from = args.init_from or (train_cfg.get("init_from") if not resume_path else None)
    if args.init_from and args.resume:
        raise ValueError("Use only one of --init-from or --resume")
    if resume_path:
        start_step = load_checkpoint(path=str(resume_path), model=model, optimizer=optimizer, device=device)
    elif init_from:
        start_step = load_init_weights(path=str(init_from), model=model, device=device, rank=rank)
    else:
        start_step = 0
    if train_cfg.get("compile", False):
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    steps_per_epoch = len(train_loader) // grad_accum
    if steps_per_epoch < 1:
        raise ValueError("training loader is too small for the configured gradient accumulation")
    epochs_cfg = train_cfg.get("epochs")
    if epochs_cfg is None:
        max_steps = int(train_cfg["max_steps"])
        epochs = max(1, math.ceil(max_steps / steps_per_epoch))
    else:
        epochs = int(epochs_cfg)
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        epoch_steps = epochs * steps_per_epoch
        max_steps_cfg = train_cfg.get("max_steps")
        max_steps = epoch_steps if max_steps_cfg is None else min(int(max_steps_cfg), epoch_steps)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 500))
    save_every = int(train_cfg.get("save_every", 1000))
    eval_batches = int(train_cfg.get("eval_batches", 50))
    eval_cfg = train_cfg.get("eval", {})
    eval_mode = str(eval_cfg.get("mode", "ppl")).lower()
    out_dir = Path(train_cfg.get("out_dir", "runs/rex"))
    scaler = torch.amp.GradScaler(device="cuda", enabled=(amp_dtype is torch.float16))
    wandb_run = init_wandb(cfg, unwrap_model(model), train_cfg) if is_main_process(rank) else None

    per_gpu_batch = int(train_cfg.get("batch_size", 1))
    tokens_per_step = (
        int(data_cfg.get("block_size", rex_cfg.max_seq_len))
        * per_gpu_batch
        * grad_accum
        * world_size
    )
    if is_main_process(rank):
        print(
            f"training on {world_size} GPU(s), device={device}, starting at step {start_step}, "
            f"epochs={epochs}, steps_per_epoch={steps_per_epoch}, target_steps={max_steps}, "
            f"global_batch_tokens={tokens_per_step}"
        )
    model.train()
    step = start_step
    pbar = tqdm(total=max_steps, initial=min(start_step, max_steps), desc="train", disable=not is_main_process(rank))
    last_log = time.time()

    start_epoch = min(start_step // steps_per_epoch, epochs)
    for epoch in range(start_epoch, epochs):
        if step >= max_steps:
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar.set_description(f"epoch {epoch + 1}/{epochs}")
        data_iter = iter(train_loader)
        for _ in range(steps_per_epoch):
            if step >= max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(grad_accum):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(input_ids, labels=labels)
                    loss = out["loss"] / grad_accum
                scaler.scale(loss).backward()
                total_loss += float(loss.item())

            lr = resolve_lr(step, train_cfg, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            pbar.update(1)
            if step % log_every == 0:
                elapsed = max(1e-6, time.time() - last_log)
                toks_per_sec = tokens_per_step * log_every / elapsed
                last_log = time.time()
                pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{lr:.2e}", toks_s=f"{toks_per_sec:.0f}")
                if wandb_run is not None:
                    log_payload = {
                        "train/loss": total_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                        "train/tokens_per_second": toks_per_sec,
                        "train/tokens_seen": step * tokens_per_step,
                    }
                    if "step_losses" in out:
                        for idx, step_loss in enumerate(out["step_losses"].tolist()):
                            log_payload[f"train/step{idx + 1}_loss"] = step_loss
                    if "exit_probs" in out:
                        for idx, exit_prob in enumerate(out["exit_probs"].tolist()):
                            log_payload[f"train/exit_prob_step{idx + 1}"] = exit_prob
                    if "kl_uniform" in out:
                        log_payload["train/kl_uniform"] = float(out["kl_uniform"].item())
                    if "task_loss" in out:
                        log_payload["train/task_loss"] = float(out["task_loss"].item())
                    wandb_run.log(log_payload, step=step)
            if eval_every > 0 and step % eval_every == 0:
                if is_main_process(rank):
                    run_validation(
                        eval_mode=eval_mode,
                        eval_cfg=eval_cfg,
                        model=unwrap_model(model),
                        val_loader=val_loader,
                        cfg=cfg,
                        tokenizer=tokenizer,
                        device=device,
                        amp_dtype=amp_dtype,
                        eval_batches=eval_batches,
                        step=step,
                        out_dir=out_dir,
                        wandb_run=wandb_run,
                    )
                if distributed:
                    dist.barrier()
            if save_every > 0 and step % save_every == 0 and is_main_process(rank):
                save_checkpoint(path=out_dir / f"ckpt_step{step}.pt", model=model, optimizer=optimizer, step=step, cfg=cfg)
            if distributed and save_every > 0 and step % save_every == 0:
                dist.barrier()

    if is_main_process(rank):
        save_checkpoint(path=out_dir / "ckpt_final.pt", model=model, optimizer=optimizer, step=step, cfg=cfg)
    if distributed:
        dist.barrier()
    if wandb_run is not None:
        wandb_run.finish()
    pbar.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
