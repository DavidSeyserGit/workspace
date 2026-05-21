"""REX: a recursive decoder-only Transformer language model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


from ouro_loss import exit_distribution, ouro_loop_loss


@dataclass
class RexConfig:
    vocab_size: int = 49_152
    max_seq_len: int = 2048
    d_model: int = 1536
    n_heads: int = 16
    n_kv_heads: int | None = None
    n_layers: int = 8
    recurrence_steps: int = 2
    ffn_dim: int = 3968
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_step_embeddings: bool = True
    use_sandwich_norm: bool = True
    use_per_step_loss: bool = False
    use_exit_gates: bool = False
    exit_gate_beta: float = 0.05
    local_block_size: int | None = None
    rope_theta: float = 50_000.0
    early_exit_threshold: float | None = None
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RexConfig":
        fields = {name for name in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in fields})
        if cfg.n_kv_heads is None:
            cfg.n_kv_heads = cfg.n_heads
        return cfg

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.local_block_size is not None and self.local_block_size < 0:
            raise ValueError("local_block_size must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    batch, n_kv, seq_len, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, n_kv, n_rep, seq_len, head_dim)
    return hidden.reshape(batch, n_kv * n_rep, seq_len, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 50_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


def _safe_torch_load(path: str | Path, map_location: str | torch.device | None) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class CausalSelfAttention(nn.Module):
    """Grouped-query attention with optional non-overlapping block-local masking."""

    def __init__(self, cfg: RexConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        if cfg.n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = cfg.n_heads // n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for rotary embeddings")
        kv_dim = n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.local_block_size = int(cfg.local_block_size or 0)
        self.rotary = RotaryEmbedding(self.head_dim, cfg.max_seq_len, base=cfg.rope_theta)
        self._attn_bias_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def _block_local_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (seq_len, str(device), dtype)
        cached = self._attn_bias_cache.get(key)
        if cached is not None:
            return cached
        positions = torch.arange(seq_len, device=device)
        block_ids = positions // self.local_block_size
        query_pos = positions[:, None]
        key_pos = positions[None, :]
        same_block = block_ids[:, None] == block_ids[None, :]
        causal = key_pos <= query_pos
        allowed = same_block & causal
        bias = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        bias.masked_fill_(~allowed, float("-inf"))
        self._attn_bias_cache[key] = bias
        return bias

    def forward(self, x: torch.Tensor, *, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None, cache_position: int = 0, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, width = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(cache_position + seq_len)
        cos = cos[cache_position : cache_position + seq_len]
        sin = sin[cache_position : cache_position + seq_len]
        q = apply_rotary(q, cos.to(q.device), sin.to(q.device))
        k = apply_rotary(k, cos.to(k.device), sin.to(k.device))
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        dropout_p = self.dropout if self.training else 0.0
        if past_kv is not None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
        elif self.local_block_size > 0:
            attn_bias = self._block_local_bias(seq_len, q.device, q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, width)
        y = self.out(y)
        if use_cache:
            return y, (k[:, : self.n_kv_heads], v[:, : self.n_kv_heads])
        return y


class SwiGLU(nn.Module):
    def __init__(self, cfg: RexConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RexBlock(nn.Module):
    def __init__(self, cfg: RexConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.use_sandwich_norm else None
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.post_ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.use_sandwich_norm else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.attn_norm(x)
        attn_out = self.attn(h, past_kv=past_kv, cache_position=cache_position, use_cache=use_cache)
        new_kv = None
        if use_cache:
            attn_out, new_kv = attn_out
        x = x + attn_out
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        if self.post_ffn_norm is not None:
            x = self.post_ffn_norm(x)
        if use_cache:
            return x, new_kv
        return x


class RexForCausalLM(nn.Module):
    """Decoder-only LM with a stack of blocks reused across recursive passes."""

    def __init__(self, cfg: RexConfig):
        super().__init__()
        if cfg.recurrence_steps < 1:
            raise ValueError("recurrence_steps must be >= 1")
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([RexBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if cfg.use_step_embeddings:
            self.step_embedding = nn.Parameter(torch.zeros(cfg.recurrence_steps, cfg.d_model))
        else:
            self.register_parameter("step_embedding", None)
        if cfg.use_exit_gates:
            self.exit_gate = nn.Linear(cfg.d_model, 1)
        else:
            self.register_module("exit_gate", None)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def _recurrence_forward(
        self,
        input_ids: torch.Tensor,
        *,
        normalize: bool = True,
        num_recurrence_steps: int | None = None,
        early_exit_threshold: float | None = None,
        past_key_values: list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> tuple[list[torch.Tensor], list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None, int]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len={self.cfg.max_seq_len}")

        max_steps = num_recurrence_steps or self.cfg.recurrence_steps
        max_steps = max(1, min(max_steps, self.cfg.recurrence_steps))
        hidden_states: list[torch.Tensor] = []
        step_lambdas: list[torch.Tensor] = []
        x = self.drop(self.token_embedding(input_ids))
        new_past: list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None
        if use_cache:
            new_past = past_key_values if past_key_values is not None else [[None] * self.cfg.n_layers for _ in range(max_steps)]

        exit_threshold = early_exit_threshold if early_exit_threshold is not None else self.cfg.early_exit_threshold
        exited_at = max_steps

        for step in range(max_steps):
            if self.step_embedding is not None:
                x = x + self.step_embedding[step].view(1, 1, -1)
            for layer_idx, block in enumerate(self.blocks):
                layer_past = None
                if use_cache and new_past is not None:
                    layer_past = new_past[step][layer_idx]
                block_out = block(
                    x,
                    past_kv=layer_past,
                    cache_position=cache_position,
                    use_cache=use_cache,
                )
                if use_cache:
                    x, layer_kv = block_out
                    new_past[step][layer_idx] = layer_kv
                else:
                    x = block_out
            hidden = self.final_norm(x) if normalize else x
            hidden_states.append(hidden)
            if self.exit_gate is not None and not self.training:
                token_lambda = torch.sigmoid(self.exit_gate(hidden).squeeze(-1)).mean(dim=1)
                step_lambdas.append(token_lambda)
                if exit_threshold is not None and step < max_steps - 1:
                    exit_probs = exit_distribution(torch.stack(step_lambdas, dim=1))
                    cdf = exit_probs.cumsum(dim=1)[:, step]
                    if float(cdf.mean().item()) >= exit_threshold:
                        exited_at = step + 1
                        break

        return hidden_states, new_past, exited_at

    def encode(
        self,
        input_ids: torch.Tensor,
        normalize: bool = True,
        num_recurrence_steps: int | None = None,
        early_exit_threshold: float | None = None,
    ) -> torch.Tensor:
        """Return contextual token representations for downstream tasks."""
        hidden_states, _, _ = self._recurrence_forward(
            input_ids,
            normalize=normalize,
            num_recurrence_steps=num_recurrence_steps,
            early_exit_threshold=early_exit_threshold,
        )
        return hidden_states[-1]

    @staticmethod
    def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _step_gate_lambdas(exit_gate: nn.Linear, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Per-sequence exit rates from mean token-level gate outputs."""
        return torch.stack(
            [torch.sigmoid(exit_gate(hidden).squeeze(-1)).mean(dim=1) for hidden in hidden_states],
            dim=1,
        )

    def _compute_training_loss(
        self,
        hidden_states: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        step_losses = torch.stack(
            [self._causal_lm_loss(self.lm_head(hidden), labels) for hidden in hidden_states]
        )
        aux: dict[str, torch.Tensor] = {"step_losses": step_losses.detach()}

        if self.cfg.use_exit_gates and self.exit_gate is not None:
            lambdas = self._step_gate_lambdas(self.exit_gate, hidden_states)
            exit_probs = exit_distribution(lambdas)
            loss, gate_aux = ouro_loop_loss(step_losses, exit_probs, beta=self.cfg.exit_gate_beta)
            aux.update(gate_aux)
            return loss, aux

        if self.cfg.use_per_step_loss:
            return step_losses.mean(), aux

        return step_losses[-1], aux

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        num_recurrence_steps: int | None = None,
        early_exit_threshold: float | None = None,
        past_key_values: list[list[tuple[torch.Tensor, torch.Tensor] | None]] | None = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> dict[str, torch.Tensor | None]:
        hidden_states, new_past, exited_at = self._recurrence_forward(
            input_ids,
            normalize=True,
            num_recurrence_steps=num_recurrence_steps,
            early_exit_threshold=early_exit_threshold,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        final_hidden = hidden_states[-1]
        logits = self.lm_head(final_hidden)
        loss = None
        aux: dict[str, torch.Tensor] = {}
        if labels is not None:
            loss, aux = self._compute_training_loss(hidden_states, labels)
        result: dict[str, Any] = {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_past,
            "exited_at_step": torch.tensor(exited_at),
            **aux,
        }
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        no_repeat_ngram_size: int = 0,
        *,
        num_recurrence_steps: int | None = None,
        early_exit_threshold: float | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be >= 0")
        can_cache = use_kv_cache and int(self.cfg.local_block_size or 0) == 0
        past_key_values = None
        cache_len = 0

        if can_cache and input_ids.size(1) > 0:
            prefill = self(
                input_ids,
                num_recurrence_steps=num_recurrence_steps,
                early_exit_threshold=early_exit_threshold,
                use_cache=True,
                cache_position=0,
            )
            past_key_values = prefill["past_key_values"]
            cache_len = input_ids.size(1)

        for _ in range(max_new_tokens):
            if can_cache and past_key_values is not None:
                next_in = input_ids[:, -1:]
                out = self(
                    next_in,
                    num_recurrence_steps=num_recurrence_steps,
                    early_exit_threshold=early_exit_threshold,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_len,
                )
                past_key_values = out["past_key_values"]
                cache_len += 1
                logits = out["logits"][:, -1, :]
            else:
                context = input_ids[:, -self.cfg.max_seq_len :]
                logits = self(
                    context,
                    num_recurrence_steps=num_recurrence_steps,
                    early_exit_threshold=early_exit_threshold,
                )["logits"][:, -1, :]
            logits = self._apply_no_repeat_ngram(logits, input_ids, no_repeat_ngram_size)
            if temperature < 0:
                raise ValueError("temperature must be >= 0")
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                continue
            logits = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    @staticmethod
    def _apply_no_repeat_ngram(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        if no_repeat_ngram_size <= 0:
            return logits

        logits = logits.clone()
        for batch_idx in range(input_ids.size(0)):
            banned_tokens = RexForCausalLM._get_banned_ngram_tokens(
                input_ids[batch_idx].tolist(),
                no_repeat_ngram_size,
            )
            if banned_tokens:
                logits[batch_idx, banned_tokens] = float("-inf")
        return logits

    @staticmethod
    def _get_banned_ngram_tokens(tokens: list[int], ngram_size: int) -> list[int]:
        if ngram_size == 1:
            return list(set(tokens))
        if len(tokens) < ngram_size - 1:
            return []

        prefix_to_next: dict[tuple[int, ...], set[int]] = {}
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tokens[i : i + ngram_size]
            prefix = tuple(ngram[:-1])
            prefix_to_next.setdefault(prefix, set()).add(ngram[-1])

        current_prefix = tuple(tokens[-(ngram_size - 1) :])
        return list(prefix_to_next.get(current_prefix, set()))

    def parameter_count(self, trainable_only: bool = False) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save model weights and config in a lightweight HF-style folder."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
            f.write("\n")
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
    ) -> "RexForCausalLM":
        """Load a model saved by ``save_pretrained``."""
        load_path = Path(load_directory)
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            cfg = RexConfig.from_dict(json.load(f))
        model = cls(cfg)
        state_dict = _safe_torch_load(load_path / "pytorch_model.bin", map_location)
        model.load_state_dict(state_dict, strict=strict)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
    ) -> "RexForCausalLM":
        """Load from a training checkpoint produced by ``train.py``."""
        checkpoint = _safe_torch_load(checkpoint_path, map_location)
        cfg_data = checkpoint.get("model_config")
        if cfg_data is None:
            cfg_data = checkpoint.get("config", {}).get("model")
        if cfg_data is None:
            raise ValueError("checkpoint does not contain model_config or config.model")
        model = cls(RexConfig.from_dict(cfg_data))
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=strict)
        return model


def build_model(config: dict[str, Any] | RexConfig | None = None) -> RexForCausalLM:
    if config is None:
        cfg = RexConfig()
    elif isinstance(config, RexConfig):
        cfg = config
    else:
        cfg = RexConfig.from_dict(config)
    return RexForCausalLM(cfg)
