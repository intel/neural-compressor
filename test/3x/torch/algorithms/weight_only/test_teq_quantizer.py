import copy
import unittest

import torch
import transformers

from neural_compressor.common import logger
from neural_compressor.torch.algorithms.weight_only.teq import TEQuantizer
from neural_compressor.torch.quantization import quantize


def generate_random_corpus(nsamples=32):
    meta_data = []
    for _ in range(nsamples):
        inp = torch.ones([1, 512], dtype=torch.long)
        tar = torch.ones([1, 512], dtype=torch.long)
        meta_data.append((inp, tar))
    return meta_data


def train(
    model,
    train_steps=100,
    lr=1e-3,
    warmup_ratio=0.05,
    gradient_accumulation_steps=1,
    logging_steps=10,
    betas=[0.9, 0.9],
    weight_decay=0,
    lr_scheduler_type="linear",
):
    """Train function."""
    trained_alphas_list = [torch.ones([128], requires_grad=True)]
    optimizer = torch.optim.Adam(trained_alphas_list, lr=lr, weight_decay=weight_decay, betas=betas)

    lr_scheduler = transformers.get_scheduler(  # pylint: disable=E1111
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(train_steps * warmup_ratio) // gradient_accumulation_steps,
        num_training_steps=train_steps // gradient_accumulation_steps,
    )

    logger.info("start training")
    model.train()
    global_steps = 0
    dataloader = generate_random_corpus()
    while global_steps <= train_steps:
        for inputs in dataloader:
            if isinstance(inputs, torch.Tensor):
                input_id = inputs
            elif isinstance(inputs, dict):
                input_id = inputs["input_ids"]
            else:
                input_id = inputs[0]
            output = model(input_id, labels=input_id)
            loss = output[0] / gradient_accumulation_steps
            loss.backward()
            global_steps += 1

            if global_steps % logging_steps == 0:
                logger.info("steps: {}, loss: {}".format(global_steps, loss.detach().cpu().item()))

            if global_steps % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if global_steps >= train_steps:  # pragma: no cover
                break

    logger.info("finish training")
    model.eval()
    return None


class TestTEQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.gptj.seqlen = 512

    def test_teq_detect_absorb_layers(self):
        example_inputs = torch.ones([1, 512], dtype=torch.long)
        test_input = torch.ones([1, 512], dtype=torch.long)
        model = copy.deepcopy(self.gptj)
        out0 = model(test_input)

        weight_config = {
            # 'op_name': (bit, group_size, scheme)
            "transformer.h.0.mlp.fc_in": {"bits": 8, "group_size": -1, "scheme": "sym"},
            "transformer.h.0.mlp.fc_out": {"bits": 4, "group_size": 32, "scheme": "asym"},
        }
        quantizer = TEQuantizer(quant_config=weight_config, folding=False, example_inputs=example_inputs)
        model = quantizer.quantize(copy.deepcopy(self.gptj), run_fn=train)
        out1 = model(test_input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=0.03))

    def test_teq(self):
        example_inputs = torch.ones([1, 512], dtype=torch.long)
        test_input = torch.ones([1, 512], dtype=torch.long)
        model = copy.deepcopy(self.gptj)
        out0 = model(test_input)

        weight_config = {
            # 'op_name': (bit, group_size, scheme)
            "transformer.h.0.mlp.fc_in": {"bits": 4, "group_size": -1, "scheme": "sym"},
            "transformer.h.0.mlp.fc_out": {"bits": 4, "group_size": 32, "scheme": "asym"},
        }
        # absorb_dict = {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]}
        absorb_dict = None

        quantizer = TEQuantizer(
            quant_config=weight_config, folding=False, absorb_to_layer=absorb_dict, example_inputs=example_inputs
        )
        model = quantizer.quantize(copy.deepcopy(self.gptj), run_fn=train)
        out1 = model(test_input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=0.03))

        quant_config = {
            "teq": {
                "global": {
                    "dtype": "fp32",
                },
                "local": {
                    "transformer.h.0.mlp.fc_in": {
                        "dtype": "int",
                        "bits": 8,
                        "group_size": -1,
                        "use_sym": True,
                        "folding": False,
                        # "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]},
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_in"]},
                    },
                    "transformer.h.0.mlp.fc_out": {
                        "dtype": "int",
                        "bits": 4,
                        "group_size": 32,
                        "use_sym": False,
                        "folding": False,
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_out": ["transformer.h.0.mlp.fc_out"]},
                    },
                },
            }
        }
        qdq_model = quantize(
            model=copy.deepcopy(self.gptj), quant_config=quant_config, run_fn=train, example_inputs=example_inputs
        )
        self.assertTrue(isinstance(qdq_model, torch.nn.Module))
        out2 = qdq_model(test_input)
        self.assertTrue(torch.allclose(out1[0], out2[0]))
        self.assertTrue(torch.allclose(out2[0], out0[0], atol=0.03))


if __name__ == "__main__":
    unittest.main()
