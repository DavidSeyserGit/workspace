"""Ouro-style looped LM training objectives."""

from __future__ import annotations

import torch


def exit_distribution(lambdas: torch.Tensor) -> torch.Tensor:
    """Convert per-step exit rates into a valid discrete distribution."""
    batch_size, num_steps = lambdas.shape
    survival = torch.ones(batch_size, device=lambdas.device, dtype=lambdas.dtype)
    probs = []
    for step in range(num_steps):
        if step < num_steps - 1:
            exit_prob = lambdas[:, step] * survival
            probs.append(exit_prob)
            survival = survival * (1.0 - lambdas[:, step])
        else:
            probs.append(survival)
    return torch.stack(probs, dim=1)


def kl_to_uniform(exit_probs: torch.Tensor) -> torch.Tensor:
    """KL(exit_probs || Uniform(T)) averaged over the batch."""
    num_steps = exit_probs.size(-1)
    uniform = 1.0 / num_steps
    kl = (exit_probs * (exit_probs.clamp_min(1e-8).log() - torch.log(torch.tensor(uniform, device=exit_probs.device)))).sum(dim=-1)
    return kl.mean()


def ouro_loop_loss(
    step_losses: torch.Tensor,
    exit_probs: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ouro Eq. 4: weighted step CE + beta * KL(q || uniform)."""
    task_loss = (exit_probs * step_losses.unsqueeze(0)).sum(dim=1).mean()
    kl_uniform = kl_to_uniform(exit_probs)
    loss = task_loss + float(beta) * kl_uniform
    return loss, {
        "task_loss": task_loss.detach(),
        "kl_uniform": kl_uniform.detach(),
        "exit_probs": exit_probs.detach().mean(dim=0),
    }
