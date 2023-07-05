import shutil
import torch
import unittest
import transformers
from neural_compressor import quantization, PostTrainingQuantConfig


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(30, 40)
        self.fc2 = torch.nn.Linear(40, 30)
        self.fc3 = torch.nn.Linear(30, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def eval_func(model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        input = torch.randn(3,30)
        # compute output
        output = model(input)
    return 0.0

class SimpleDataLoader():
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.randn([1, 30])


class LLMDataLoader():
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
    approach = 'weight_only'

    @classmethod
    def setUpClass(self):
        self.dataloader = SimpleDataLoader()
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            'hf-internal-testing/tiny-random-GPTJForCausalLM',
            torchscript=True,
        )
        self.llm_dataloader = LLMDataLoader()
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_RTN_quant(self):
        input = torch.randn(3,30)
        model = Model()
        out1 = model(input)

        conf = PostTrainingQuantConfig(
            approach='weight_only',
        )
        q_model = quantization.fit(model, conf)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bits': 8, # 1-8 bits 
                        'group_size': -1,  # -1 (per-channel)
                        'scheme': 'sym', 
                        'algorithm': 'RTN', 
                    },
                },
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bits': 4, # 1-8 bits 
                        'group_size': 32,  # 1 - 1024 or higher
                        'scheme': 'asym', 
                        'algorithm': 'RTN', 
                    },
                },
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_name_dict={
                'fc1':{ 	# re.match
                    "weight": {
                        'bits': 4, # 1-8 bits 
                        'group_size': 32,  # 1 - 1024 or higher
                        'scheme': 'sym', 
                        'algorithm': 'RTN', 
                    },
                },
                'fc2':{ 	# re.match
                    "weight": {
                        'bits': 3, # 1-8 bits 
                        'group_size': 16,  # 1 - 1024 or higher
                        'scheme': 'asym', 
                        'algorithm': 'RTN', 
                    },
                },
                'fc3':{ 	# re.match
                    "weight": {
                        'dtype': 'fp32',
                    },
                },
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))
        q_model.save('saved')
        from neural_compressor.utils.pytorch import load
        new_model = load('saved', model)
        out1 = new_model(input)
        self.assertTrue(torch.all(out1 == out2))

    def test_AWQ_quant(self):
        input = torch.randn(3,30)
        model = Model()
        out1 = model(input)

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bits': 4, # 1-8 bits 
                        'group_size': 32,  # -1 (per-channel)
                        'scheme': 'sym', 
                        'algorithm': 'AWQ', 
                    },
                },
            },
            op_name_dict={
                '.*lm_head':{ 	# re.match
                    "weight": {
                        'dtype': 'fp32'
                    },
                },
            },
            recipes={
                'awq_args':{'auto_scale': True, 'mse_range': True},
            },
        )
        q_model = quantization.fit(
            self.gptj, 
            conf, 
            calib_dataloader=self.llm_dataloader,
        )


if __name__ == "__main__":
    unittest.main()
