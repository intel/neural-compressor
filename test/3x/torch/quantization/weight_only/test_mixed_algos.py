import copy
from unittest.mock import patch

import pytest
import torch
import transformers

from neural_compressor.torch.quantization import GPTQConfig, RTNConfig, quantize
from neural_compressor.torch.utils import accelerator, logger

device = accelerator.current_device_name()


def run_fn(model):
    model(torch.tensor([[10, 20, 30]], dtype=torch.long).to(device))


class TestMixedTwoAlgo:
    def test_mixed_gptq_and_rtn(self):
        with patch.object(logger, "info") as mock_info:
            rtn_config = RTNConfig(white_list=[".*mlp.*"])
            gptq_config = GPTQConfig(double_quant_bits=4, white_list=[".*attn.*"])
            combined_config = rtn_config + gptq_config
            logger.info(combined_config)

            self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-GPTJForCausalLM",
                device_map=device,
            )
            self.example_inputs = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long).to(device)
            # record label for comparison
            out_original_model = self.tiny_gptj(self.example_inputs)[0]
            model = copy.deepcopy(self.tiny_gptj)
            q_model = quantize(model, combined_config, run_fn=run_fn)
            out_q_model = q_model(self.example_inputs)[0]
            rtn_log = "Start to apply rtn on the model."
            gptq_log = "Start to apply gptq on the model."
            assert rtn_log in [_call[0][0] for _call in mock_info.call_args_list]
            assert gptq_log in [_call[0][0] for _call in mock_info.call_args_list]
