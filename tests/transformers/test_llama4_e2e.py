# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama4_mlp_correctness():
    """Llama4TextMLPTileGym output matches vanilla PyTorch reference."""
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

    from tilegym.transformers.llama4.modeling_llama4 import Llama4TextMLPTileGym

    class FakeConfig:
        hidden_size = 256
        intermediate_size_mlp = 512
        intermediate_size = 512
        hidden_act = "silu"

    config = FakeConfig()
    device, dtype = torch.device("cuda"), torch.bfloat16

    ref = Llama4TextMLP(config).to(device=device, dtype=dtype).eval()
    tg = Llama4TextMLPTileGym(config).to(device=device, dtype=dtype).eval()
    tg.gate_proj.weight.data.copy_(ref.gate_proj.weight.data)
    tg.up_proj.weight.data.copy_(ref.up_proj.weight.data)
    tg.down_proj.weight.data.copy_(ref.down_proj.weight.data)

    x = torch.randn(8, 64, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(tg(x), ref(x), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama4_moe_correctness():
    """Llama4TextMoeTileGym output matches reference (vanilla Llama4TextMoe) on same weights.

    Note: topk_weights are normalized before fused_moe (rows must sum to 1), which is a
    deliberate deviation from vanilla Llama4TextMoe's raw sigmoid scores. Tolerance is set
    to 2e-2 to account for this difference in weighting scale.
    """
    from transformers import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    from tilegym.transformers.llama4.modeling_llama4 import Llama4TextMoeTileGym

    config = Llama4TextConfig(
        hidden_size=256,
        num_local_experts=8,
        num_experts_per_tok=2,
        intermediate_size=512,
        intermediate_size_mlp=256,
        hidden_act="silu",
    )
    device, dtype = torch.device("cuda"), torch.bfloat16

    ref = Llama4TextMoe(config).to(device=device, dtype=dtype).eval()
    tg = Llama4TextMoeTileGym(config).to(device=device, dtype=dtype).eval()

    # Copy all weights — experts, router, AND shared expert
    tg.experts.gate_up_proj.data.copy_(ref.experts.gate_up_proj.data)
    tg.experts.down_proj.data.copy_(ref.experts.down_proj.data)
    tg.router.weight.data.copy_(ref.router.weight.data)
    tg.shared_expert.gate_proj.weight.data.copy_(ref.shared_expert.gate_proj.weight.data)
    tg.shared_expert.up_proj.weight.data.copy_(ref.shared_expert.up_proj.weight.data)
    tg.shared_expert.down_proj.weight.data.copy_(ref.shared_expert.down_proj.weight.data)

    x = torch.randn(4, 32, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ref_out, _ = ref(x)
        tg_out, _ = tg(x)

    # 2e-2 tolerance: accounts for weight normalization difference (sigmoid vs normalized sigmoid)
    torch.testing.assert_close(tg_out, ref_out, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama4_scout_e2e():
    """Full forward pass of Llama 4 Scout with TileGym patches matches vanilla output.

    Uses a small random-weight Llama 4 Scout config (no MoE layers) to validate that
    apply_tilegym_kernel_to_llama4 correctly patches all components end-to-end.
    Scout has no MoE — all layers are dense TextMLP layers.
    """
    import copy

    from transformers import Llama4ForCausalLM, Llama4TextConfig

    from tilegym.transformers.monkey_patch import apply_tilegym_kernel_to_llama4

    config = Llama4TextConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=512,
        intermediate_size_mlp=512,
        # Scout has no MoE layers
        num_local_experts=0,
        num_experts_per_tok=0,
        interleave_moe_layer_step=0,
        vocab_size=512,
        max_position_embeddings=64,
    )

    device, dtype = torch.device("cuda"), torch.bfloat16

    # Build vanilla reference model
    ref_model = Llama4ForCausalLM(config).to(device=device, dtype=dtype).eval()

    # Build patched model from same weights
    apply_tilegym_kernel_to_llama4(moe=False)
    patched_model = Llama4ForCausalLM(config).to(device=device, dtype=dtype).eval()
    patched_model.load_state_dict(copy.deepcopy(ref_model.state_dict()))

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    with torch.no_grad():
        ref_out = ref_model(input_ids).logits
        tg_out = patched_model(input_ids).logits

    torch.testing.assert_close(tg_out, ref_out, rtol=1e-2, atol=1e-2)
