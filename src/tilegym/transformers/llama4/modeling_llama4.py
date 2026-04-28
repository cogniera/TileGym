# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from tilegym.ops import get_fused_swiglu_module
from tilegym.ops.fused_mlp import PartiallyFusedSwiGLUMLP
from tilegym.ops.moe_interface import fused_moe


class Llama4TextMLPTileGym(PartiallyFusedSwiGLUMLP):
    """Drop-in replacement for Llama4TextMLP.

    Subclasses PartiallyFusedSwiGLUMLP to pass the correct config field.
    Llama4 dense layers use config.intermediate_size_mlp, not config.intermediate_size.
    """

    def __init__(self, config, intermediate_size=None):
        if intermediate_size is None:
            intermediate_size = config.intermediate_size_mlp
        super().__init__(config, intermediate_size=intermediate_size)


class Llama4TextMoeTileGym(nn.Module):
    """Drop-in replacement for Llama4TextMoe.

    Reuses TileGym's fused_moe kernel with transposed weight layout and normalized routing.
    """

    def __init__(self, config):
        super().__init__()
        from transformers.models.llama4.modeling_llama4 import Llama4Router
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts

        # Keep original HF modules so checkpoint weights load correctly
        self.experts = Llama4TextExperts(config)
        self.router = Llama4Router(config)

        # Shared expert uses separate gate/up/down projections with intermediate_size (not _mlp)
        FusedSwiGLUMLP = get_fused_swiglu_module()
        self.shared_expert = FusedSwiGLUMLP(config=config, intermediate_size=config.intermediate_size)

        # Lazily transposed weight views — note: NOT named _init_weights (reserved by nn.Module)
        self._w1 = None  # [E, 2*expert_dim, hidden_size]
        self._w2 = None  # [E, hidden_size, expert_dim]

    def _build_fused_weights(self):
        """Lazily transpose expert weights to TileGym's expected layout.

        Called on first forward to avoid transposing on CPU before model is on GPU.
        """
        if self._w1 is not None:
            return
        # gate_up_proj: [E, hidden_size, 2*expert_dim] → [E, 2*expert_dim, hidden_size]
        self._w1 = self.experts.gate_up_proj.transpose(1, 2).contiguous()
        # down_proj:    [E, expert_dim, hidden_size]   → [E, hidden_size, expert_dim]
        self._w2 = self.experts.down_proj.transpose(1, 2).contiguous()

    def forward(self, hidden_states):
        """Forward pass with fused MoE and shared expert.

        Note: topk_weights are normalized to sum to 1 per token, which is a deliberate
        deviation from vanilla Llama4TextMoe's raw sigmoid scores. This is required by
        fused_moe but changes the weighting semantics slightly.
        """
        self._build_fused_weights()
        residual = hidden_states
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # Llama4Router returns sigmoid scores: topk positions have values, rest are -inf
        router_scores, router_logits = self.router(hidden_states)
        topk_weights, topk_ids = torch.topk(router_scores, self.top_k, dim=-1)

        # fused_moe requires normalized weights (rows sum to 1) — sigmoid scores do not.
        # This is a deliberate deviation from vanilla Llama4TextMoe which uses raw sigmoid
        # scores for weighting. Normalization preserves relative expert importance.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = fused_moe(
            hidden_states,
            w1=self._w1,
            w2=self._w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        out = out.reshape(residual.shape)
        out = out + self.shared_expert(residual)
        return out, router_logits
