import pytest
import torch
import transformers

from neural_compressor.torch.algorithms.weight_only.autoround import AutoRoundQuantizer, get_autoround_default_run_fn
from neural_compressor.torch.quantization import AutoRoundConfig, quantize
from neural_compressor.torch.utils import logger

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


@pytest.fixture(scope="module")
def gpt_j():
    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


@pytest.mark.skipif(not auto_round_installed, reason="auto_round module is not installed")
class TestAutoRound:
    @staticmethod
    @pytest.fixture(scope="class", autouse=True)
    def gpt_j_model(gpt_j):
        yield gpt_j

    def setup_method(self, method):
        logger.info(f"Running TestAutoRound test: {method.__name__}")

    def test_autoround(self, gpt_j_model):
        inp = torch.ones([1, 10], dtype=torch.long)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )

        out1 = gpt_j_model(inp)
        quant_config = AutoRoundConfig(n_samples=20, seqlen=10, iters=10, scale_dtype="fp32")
        logger.info(f"Test AutoRound with config {quant_config}")

        qdq_model = quantize(
            model=gpt_j_model,
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

        q_model = qdq_model
        out2 = q_model(inp)
        assert torch.allclose(out1[0], out2[0], atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]

    def test_new_api(self, gpt_j_model):
        inp = torch.ones([1, 10], dtype=torch.long)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )

        out1 = gpt_j_model(inp)

        run_fn = get_autoround_default_run_fn
        run_args = (
            tokenizer,
            "NeelNanda/pile-10k",
            20,
            10,
        )
        weight_config = {
            "*": {
                "data_type": "int",
                "bits": 4,
                "group_size": 32,
                "sym": False,
            }
        }
        quantizer = AutoRoundQuantizer(weight_config=weight_config)
        fp32_model = gpt_j_model

        # quantizer execute
        model = quantizer.prepare(model=fp32_model)
        run_fn(model, *run_args)
        q_model = quantizer.convert(model)

        out2 = q_model(inp)
        assert torch.allclose(out1[0], out2[0], atol=1e-1)
        assert "transformer.h.0.attn.k_proj" in q_model.autoround_config.keys()
        assert "scale" in q_model.autoround_config["transformer.h.0.attn.k_proj"].keys()
        assert torch.float32 == q_model.autoround_config["transformer.h.0.attn.k_proj"]["scale_dtype"]
