import habana_frameworks.torch.core as htcore
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from neural_compressor.torch.algorithms.mixed_low_precision.custom_methods.quarot import rotate


class RotationOptions:
    def __init__(self, rotate_weights=True, rotate_values=False, rotate_mlp=True):
        self.rotate_weights = rotate_weights
        self.rotate_values = rotate_values
        self.rotate_mlp = rotate_mlp


def get_model():
    config_dict = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 2,
        "num_key_value_heads": 32,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "vocab_size": 32000,
    }

    config = LlamaConfig(**config_dict)
    model = LlamaForCausalLM(config)
    return model


def test_quarot():
    options = RotationOptions(rotate_weights=True, rotate_values=False, rotate_mlp=True)
    model = get_model()
    model.model.layers = model.model.layers[:2]
    input = torch.ones((1, 5), dtype=int).to("hpu")
    with torch.no_grad():
        output_logits = model(input).logits.cpu()
    rotate(model, options)
    with torch.no_grad():
        htcore.mark_step()
        output_rotated_logits = model(input).logits.cpu()
    assert torch.allclose(output_logits, output_rotated_logits, atol=1)
