"""Hugging Face configuration for REX."""

from __future__ import annotations

from transformers import PretrainedConfig


class RexConfig(PretrainedConfig):
    model_type = "rex"

    def __init__(
        self,
        vocab_size: int = 49_152,
        max_seq_len: int = 2048,
        d_model: int = 1536,
        n_heads: int = 16,
        n_kv_heads: int | None = None,
        n_layers: int = 8,
        recurrence_steps: int = 2,
        ffn_dim: int = 3968,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        tie_embeddings: bool = True,
        use_step_embeddings: bool = True,
        use_per_step_loss: bool = False,
        use_exit_gates: bool = False,
        exit_gate_beta: float = 0.05,
        use_sandwich_norm: bool = True,
        local_block_size: int | None = None,
        rope_theta: float = 50_000.0,
        early_exit_threshold: float | None = None,
        initializer_range: float = 0.02,
        tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M",
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.n_layers = n_layers
        self.recurrence_steps = recurrence_steps
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.use_step_embeddings = use_step_embeddings
        self.use_per_step_loss = use_per_step_loss
        self.use_exit_gates = use_exit_gates
        self.exit_gate_beta = exit_gate_beta
        self.use_sandwich_norm = use_sandwich_norm
        self.local_block_size = local_block_size
        self.rope_theta = rope_theta
        self.early_exit_threshold = early_exit_threshold
        self.initializer_range = initializer_range
        self.tokenizer_name = tokenizer_name

    def to_core_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "recurrence_steps": self.recurrence_steps,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "norm_eps": self.norm_eps,
            "tie_embeddings": self.tie_embeddings,
            "use_step_embeddings": self.use_step_embeddings,
            "use_per_step_loss": self.use_per_step_loss,
            "use_exit_gates": self.use_exit_gates,
            "exit_gate_beta": self.exit_gate_beta,
            "use_sandwich_norm": self.use_sandwich_norm,
            "local_block_size": self.local_block_size,
            "rope_theta": self.rope_theta,
            "early_exit_threshold": self.early_exit_threshold,
            "initializer_range": self.initializer_range,
        }
