"""Hugging Face model wrapper for REX."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_rex import RexConfig
from model import RexConfig as CoreRexConfig
from model import RexForCausalLM as CoreRexForCausalLM


class RexForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RexConfig
    base_model_prefix = "rex"
    supports_gradient_checkpointing = False
    _tied_weights_keys = ["rex.lm_head.weight"]

    def __init__(self, config: RexConfig):
        super().__init__(config)
        self.rex = CoreRexForCausalLM(CoreRexConfig.from_dict(config.to_core_dict()))

    def get_input_embeddings(self) -> nn.Module:
        return self.rex.token_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.rex.token_embedding = value
        if self.rex.cfg.tie_embeddings:
            self.rex.lm_head.weight = self.rex.token_embedding.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.rex.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.rex.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": input_ids[:, -self.config.max_seq_len :]}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        num_recurrence_steps: int | None = None,
        early_exit_threshold: float | None = None,
        **_: Any,
    ) -> CausalLMOutputWithPast:
        out = self.rex(
            input_ids=input_ids,
            labels=labels,
            num_recurrence_steps=num_recurrence_steps,
            early_exit_threshold=early_exit_threshold,
            past_key_values=past_key_values,
            use_cache=bool(use_cache),
        )
        return CausalLMOutputWithPast(
            loss=out["loss"],
            logits=out["logits"],
            past_key_values=out.get("past_key_values"),
        )
