import copy
import shutil
import unittest

import torch
import transformers

from neural_compressor.common import logger
from neural_compressor.torch.quantization import convert, prepare, quantize
from neural_compressor.torch.utils import accelerator

device = accelerator.current_device_name()


def generate_random_corpus(nsamples=32):
    meta_data = []
    for _ in range(nsamples):
        inp = torch.ones([1, 512], dtype=torch.long).to(device)
        tar = torch.ones([1, 512], dtype=torch.long).to(device)
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
    trained_alphas_list = [torch.ones([128], requires_grad=True).to(device)]
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
            device_map=device,
        )
        self.gptj.seqlen = 512
        self.example_inputs = torch.ones([1, 512], dtype=torch.long).to(device)

        self.quant_config = {
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
                        "folding": True,
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]},
                    },
                    "transformer.h.0.mlp.fc_out": {
                        "dtype": "int",
                        "bits": 4,
                        "group_size": 32,
                        "use_sym": False,
                        "folding": True,
                        "absorb_to_layer": {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]},
                    },
                },
            }
        }

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    def test_teq(self):
        test_input = torch.ones([1, 512], dtype=torch.long)
        model = copy.deepcopy(self.gptj)
        out0 = model(test_input)
        prepared_model = prepare(model, quant_config=self.quant_config, example_inputs=self.example_inputs)
        train(prepared_model)
        qdq_model = convert(prepared_model)
        assert qdq_model is not None, "Quantization failed!"
        self.assertTrue(isinstance(qdq_model, torch.nn.Module))
        out1 = qdq_model(test_input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=0.03))

    def test_save_and_load(self):
        fp32_model = copy.deepcopy(self.gptj)
        prepared_model = prepare(fp32_model, quant_config=self.quant_config, example_inputs=self.example_inputs)
        train(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(self.example_inputs)[0]

        from neural_compressor.torch.quantization import load

        # loading compressed model
        loaded_model = load("saved_results", copy.deepcopy(self.gptj))
        loaded_out = loaded_model(self.example_inputs)[0]
        assert torch.allclose(inc_out, loaded_out), "Unexpected result. Please double check."

    def test_teq_with_quantize_API(self):
        example_inputs = torch.ones([1, 512], dtype=torch.long)
        test_input = torch.ones([1, 512], dtype=torch.long)

        # prepare + convert API
        prepared_model = prepare(
            copy.deepcopy(self.gptj), quant_config=self.quant_config, example_inputs=example_inputs
        )
        train(prepared_model)
        qdq_model = convert(prepared_model)
        self.assertTrue(isinstance(qdq_model, torch.nn.Module))
        out1 = qdq_model(test_input)

        # quantize API
        qdq_model = quantize(
            model=copy.deepcopy(self.gptj), quant_config=self.quant_config, run_fn=train, example_inputs=example_inputs
        )
        self.assertTrue(isinstance(qdq_model, torch.nn.Module))
        out2 = qdq_model(test_input)

        # compare the results of calling `convert` + `prepare` and calling `quantize`
        assert torch.all(
            out1[0].eq(out2[0])
        ), "The results of calling `convert` + `prepare` and calling `quantize` should be equal."
