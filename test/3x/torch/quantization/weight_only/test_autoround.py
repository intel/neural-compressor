import unittest

import torch
import transformers

from neural_compressor.torch.quantization import AUTOROUNDConfig, quantize
from neural_compressor.torch.utils import logger

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


def get_gpt_j():
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


@unittest.skipIf(not auto_round_installed, "auto_round module is not installed")
class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = get_gpt_j()

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestAutoRound test: {self.id()}")

    def test_autoround(self):
        """ "
        "n_samples": 20,
        "amp": False,
        "seq_len": 10,
        "iters": 10,
        "scale_dtype": "fp32",
        "device": "cpu","""
        inp = torch.ones([1, 10], dtype=torch.long)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )

        out1 = self.gptj(inp)
        quant_config = AUTOROUNDConfig(n_samples=20, amp=False, seqlen=10, iters=10, scale_dtype="fp32", device="cpu")
        logger.info(f"Test AutoRound with config {quant_config}")
        from neural_compressor.torch.algorithms.weight_only.autoround import get_autoround_default_run_fn

        qdq_model = quantize(
            model=self.gptj,
            quant_config=quant_config,
            run_fn=get_autoround_default_run_fn,
            run_args=(
                tokenizer,
                "NeelNanda/pile-10k",
                20,
                10,
            ),
        )
        """run_args of get_autoround_default_run_fn:
            tokenizer,
            dataset_name="NeelNanda/pile-10k",
            n_samples=512,
            seqlen=2048,
            seed=42,
            bs=8,
            dataset_split: str = "train",
            dataloader=None,
        """

        out2 = qdq_model(inp)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-1))

        q_model = qdq_model
        out2 = q_model(inp)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-01))
        self.assertTrue("transformer.h.0.attn.k_proj" in q_model.autoround_config.keys())
        self.assertTrue("scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys())
        self.assertTrue(torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"])


if __name__ == "__main__":
    unittest.main()
