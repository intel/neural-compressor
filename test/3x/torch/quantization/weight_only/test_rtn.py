import copy
import unittest

import transformers

from neural_compressor.torch.quantization import RTNConfig, get_default_rtn_config, quantize


class TestRTNQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )

    @classmethod
    def tearDownClass(self):
        pass

    def test_export_compressed_model(self):
        model = copy.deepcopy(self.tiny_gptj)
        quant_config = RTNConfig(export_compressed_model=True)
        model = quantize(model, quant_config)
        print(model)


if __name__ == "__main__":
    unittest.main()
