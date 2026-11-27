# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit and integration tests for ``neural_compressor.torch.export`` (Phase 3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from neural_compressor.torch.export import (
    dequantize_compressed_tensors_layer,
    export_hf2compressored_model,
    export_hf_checkpoint,
    pack_linear_tensors,
)
from neural_compressor.torch.quantization import FakeQuantConfig, prepare_qat
from neural_compressor.torch.quantization.fake_quant import (
    QuantizerConfig,
    QuantSchemeRule,
    TensorQuantizer,
    calibrate_static_amax,
    ensure_peft_torchao_compat,
    save_quantizer_state,
)

ensure_peft_torchao_compat()


def _make_tiny_qwen2(seed: int = 7) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16)


def _load_all_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.is_file():
        idx = json.loads(idx_file.read_text(encoding="utf-8"))
        shards = sorted(set(idx["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    tensors: dict[str, torch.Tensor] = {}
    for s in shards:
        with safe_open(str(model_dir / s), framework="pt", device="cpu") as h:
            for k in h.keys():
                tensors[k] = h.get_tensor(k)
    return tensors


@pytest.mark.parametrize(
    "fmt,dynamic_in,set_amax",
    [
        ("mxfp4", True, False),
        ("nvfp4", True, False),
        ("nvfp4", True, True),
        ("mxfp8", True, False),
        ("fp8_e4m3", False, True),
        ("fp8_e4m3", True, False),
    ],
)
def test_pack_linear_tensors_zero_ulp_vs_fake_quant(fmt: str, dynamic_in: bool, set_amax: bool) -> None:
    torch.manual_seed(123)
    weight = torch.randn(32, 64, dtype=torch.bfloat16) * 2.5
    bias = torch.randn(32, dtype=torch.bfloat16) * 0.1

    w_dyn = fmt != "fp8_e4m3"
    wq = TensorQuantizer(QuantizerConfig(format=fmt, dynamic=w_dyn))
    iq = TensorQuantizer(QuantizerConfig(format=fmt, dynamic=dynamic_in))
    if not w_dyn or set_amax:
        wq.set_amax(float(weight.float().abs().amax().item()))
    if not dynamic_in or set_amax:
        iq.set_amax(3.75)

    tensors, spec = pack_linear_tensors("model.layers.0.proj", weight, bias, wq, iq)
    assert spec is not None
    assert "model.layers.0.proj.bias" in tensors

    w_deq = dequantize_compressed_tensors_layer(
        "model.layers.0.proj",
        tensors,
        spec.format,
        original_shape=tuple(weight.shape),
        dtype=torch.bfloat16,
    )
    w_fq = wq(weight)
    assert torch.equal(w_deq, w_fq), f"max_abs={(w_deq.float() - w_fq.float()).abs().max().item()}"


def test_pack_uncalibrated_static_fp8_raises() -> None:
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    wq = TensorQuantizer(QuantizerConfig(format="fp8_e4m3", dynamic=False))
    with pytest.raises(RuntimeError, match="has no calibrated amax"):
        pack_linear_tensors("layer.q_proj", weight, None, wq, None)


def test_export_hybrid_mxfp4_fp8_in_memory_and_streaming_checkpoint(tmp_path: Path) -> None:
    base_model = _make_tiny_qwen2()
    calib_batches = [{"input_ids": torch.randint(0, 128, (2, 16))} for _ in range(2)]
    qcfg = FakeQuantConfig.mxfp4_fp8_hybrid()

    mem_model = copy.deepcopy(base_model)
    prepare_qat(mem_model, qcfg)
    calibrate_static_amax(mem_model, dataloader=calib_batches)

    # Save dense checkpoint + quantizer_state.json for streaming Path B comparison
    ckpt_dir = tmp_path / "saved_ckpt"
    ckpt_dir.mkdir(parents=True)
    dense_sd = {
        k: v.detach().cpu().contiguous()
        for k, v in mem_model.state_dict().items()
        if ".weight_quantizer." not in k and ".input_quantizer." not in k and ".output_quantizer." not in k
    }
    save_file(dense_sd, str(ckpt_dir / "model.safetensors"))
    mem_model.config.save_pretrained(ckpt_dir)
    qstate_path = save_quantizer_state(mem_model, ckpt_dir / "quantizer_state.json")

    out_mem = export_hf_checkpoint(mem_model, tmp_path / "export_mem", export_format="compressed_tensors")
    out_stream = export_hf_checkpoint(
        ckpt_dir,
        tmp_path / "export_stream",
        export_format="compressed_tensors",
        quant_config=qcfg,
        quant_state_path=qstate_path,
    )

    cfg_mem = json.loads((out_mem / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    cfg_stream = json.loads((out_stream / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    assert cfg_mem == cfg_stream
    assert cfg_mem["quant_method"] == "compressed-tensors"
    assert cfg_mem["format"] == "mixed-precision"
    assert cfg_mem["quantization_status"] == "compressed"
    assert "lm_head" in cfg_mem["ignore"]

    sd_mem = _load_all_safetensors(out_mem)
    sd_stream = _load_all_safetensors(out_stream)
    assert set(sd_mem.keys()) == set(sd_stream.keys())
    for k in sd_mem:
        assert torch.equal(sd_mem[k], sd_stream[k]), f"Mismatch on tensor {k}"

    # Verify MLP uses packed uint8 + E8M0 uint8, Attention uses FP8 + [1] FP32 weight_scale/input_scale
    assert sd_mem["model.layers.0.mlp.gate_proj.weight_packed"].dtype == torch.uint8
    assert sd_mem["model.layers.0.mlp.gate_proj.weight_packed"].shape == (128, 32)
    assert sd_mem["model.layers.0.mlp.gate_proj.weight_scale"].dtype == torch.uint8
    assert sd_mem["model.layers.0.mlp.gate_proj.weight_scale"].shape == (128, 2)
    assert sd_mem["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert sd_mem["model.layers.0.self_attn.q_proj.weight_scale"].shape == (1,)
    assert sd_mem["model.layers.0.self_attn.q_proj.input_scale"].shape == (1,)


def test_export_peft_lora_in_memory_and_streaming_adapter(tmp_path: Path) -> None:
    base_model = _make_tiny_qwen2(seed=21)
    base_dir = tmp_path / "base_hf"
    base_model.save_pretrained(base_dir)

    peft_model = get_peft_model(
        copy.deepcopy(base_model),
        LoraConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj", "down_proj"], bias="none"),
        autocast_adapter_dtype=False,
    )
    # Non-zero LoRA B so merge materially modifies weights
    for mod in peft_model.modules():
        if hasattr(mod, "lora_B") and "default" in mod.lora_B:
            nn.init.normal_(mod.lora_B["default"].weight, std=0.05)

    adapter_dir = tmp_path / "lora_adapter"
    peft_model.save_pretrained(adapter_dir)

    qcfg = FakeQuantConfig(
        config_groups=[
            QuantSchemeRule(
                targets=["*gate_proj*", "*up_proj*", "*down_proj*"],
                weights="mxfp4",
                input_activations="mxfp4",
            )
        ],
        ignore=["lm_head"],
    )
    prepare_qat(peft_model, qcfg)

    out_mem = export_hf_checkpoint(peft_model, tmp_path / "lora_ct_mem", export_format="compressed_tensors")
    out_stream = export_hf_checkpoint(
        adapter_dir,
        tmp_path / "lora_ct_stream",
        export_format="compressed_tensors",
        quant_config=qcfg,
        base_model_path=base_dir,
    )

    sd_mem = _load_all_safetensors(out_mem)
    sd_stream = _load_all_safetensors(out_stream)
    assert set(sd_mem.keys()) == set(sd_stream.keys())
    for k in sd_mem:
        assert torch.equal(sd_mem[k], sd_stream[k]), f"LoRA export mismatch on {k}"


def test_streaming_vlm_surgery_and_sharding(tmp_path: Path) -> None:
    text_model = _make_tiny_qwen2(seed=42)
    qcfg = FakeQuantConfig.mxfp4()
    prepare_qat(text_model, qcfg)

    # Construct a mock multimodal VLM checkpoint directory with visual.* and mtp.* weights
    vlm_base = tmp_path / "mock_vlm_base"
    vlm_base.mkdir(parents=True)
    vlm_tensors: dict[str, torch.Tensor] = {}
    for k, v in text_model.state_dict().items():
        if ".weight_quantizer." in k or ".input_quantizer." in k or ".output_quantizer." in k:
            continue
        mapped = k if k == "lm_head.weight" else f"model.language_model.{k[len('model.'):]}"
        vlm_tensors[mapped] = v.detach().cpu().contiguous()
    vlm_tensors["model.visual.blocks.0.mlp.linear_fc1.weight"] = torch.randn(32, 32, dtype=torch.bfloat16)
    vlm_tensors["mtp.layers.0.self_attn.q_proj.weight"] = torch.randn(32, 32, dtype=torch.bfloat16)
    save_file(vlm_tensors, str(vlm_base / "model.safetensors"))
    vlm_cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "vision_config": {"hidden_size": 32},
        "text_config": text_model.config.to_dict(),
    }
    (vlm_base / "config.json").write_text(json.dumps(vlm_cfg), encoding="utf-8")
    (vlm_base / "processor_config.json").write_text('{"processor_class": "Qwen3VLProcessor"}', encoding="utf-8")

    # Force multi-shard output via tiny max_shard_size="32KB"
    out_vlm = export_hf_checkpoint(
        text_model,
        tmp_path / "exported_vlm_ct",
        export_format="compressed_tensors",
        base_model_path=vlm_base,
        keep_vision=True,
        max_shard_size="32KB",
    )

    assert (out_vlm / "model.safetensors.index.json").is_file()
    assert (out_vlm / "processor_config.json").is_file()
    exported_cfg = json.loads((out_vlm / "config.json").read_text(encoding="utf-8"))
    assert exported_cfg["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    q_cfg = exported_cfg["quantization_config"]
    targets = q_cfg["config_groups"]["group_0"]["targets"]
    assert all(t.startswith("model.language_model.layers.") for t in targets)
    assert "re:.*visual.*" in q_cfg["ignore"]
    assert "re:.*mtp.*" in q_cfg["ignore"]

    sd = _load_all_safetensors(out_vlm)
    assert "model.language_model.layers.0.mlp.gate_proj.weight_packed" in sd
    assert torch.equal(sd["model.visual.blocks.0.mlp.linear_fc1.weight"], vlm_tensors["model.visual.blocks.0.mlp.linear_fc1.weight"])
    assert torch.equal(sd["mtp.layers.0.self_attn.q_proj.weight"], vlm_tensors["mtp.layers.0.self_attn.q_proj.weight"])


def test_legacy_export_hf2compressored_model_compat(tmp_path: Path) -> None:
    model = _make_tiny_qwen2(seed=9)
    prepare_qat(model, FakeQuantConfig.mxfp4())
    out = export_hf2compressored_model(model, tmp_path / "legacy_ct")
    assert (Path(out) / "config.json").is_file()
    assert (Path(out) / "model.safetensors").is_file()


def test_export_autodiscovers_qat_trainer_quant_config_json(tmp_path: Path) -> None:
    model = _make_tiny_qwen2(seed=11)
    ckpt_dir = tmp_path / "qat_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)

    qcfg = FakeQuantConfig(
        config_groups=[
            QuantSchemeRule(
                targets=[r"re:.*layers\.0\.self_attn\.q_proj$"],
                weights={"format": "fp8_e4m3", "dynamic": False},
                input_activations={"format": "fp8_e4m3", "dynamic": False},
            )
        ]
    )
    qcfg.to_json_file(ckpt_dir / "quant_config.json")
    wrapped_state = {
        "version": 1,
        "quantizers": {
            "base_model.model.model.layers.0._fsdp_wrapped_module.self_attn.q_proj.base_layer.weight_quantizer": {
                "amax_hex": (2.24).hex()
            },
            "base_model.model.model.layers.0._fsdp_wrapped_module.self_attn.q_proj.base_layer.input_quantizer": {
                "amax_hex": (4.48).hex()
            },
        },
    }
    (ckpt_dir / "quantizer_state.json").write_text(json.dumps(wrapped_state), encoding="utf-8")

    out_dir = export_hf_checkpoint(ckpt_dir, tmp_path / "exported_auto_ct", export_format="compressed_tensors")
    sd = _load_all_safetensors(out_dir)
    assert sd["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        sd["model.layers.0.self_attn.q_proj.input_scale"],
        torch.tensor([4.48 / 448.0], dtype=torch.float32),
    )

