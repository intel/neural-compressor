import shutil
import torch
import unittest
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


class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
    approach = 'weight_only'

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_RTN_func(self):
        # TODO
        pass

    def test_RTN_quant(self):
        input = torch.randn(3,30)
        model = Model()
        out1 = model(input)

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bit': [8], # 1-8 bit 
                        'group_size': [128],  # 1 - 1024 or higher
                        'scheme': ['sym'], 
                        'algorithm': ['RTN'], 
                    },
                },
            },
            recipes={
                'gptq_args':{'percdamp': 0.01},
                'awq_args':{'alpha': 'auto', 'clip': True},
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bit': [4], # 1-8 bit 
                        'group_size': [32],  # 1 - 1024 or higher
                        'scheme': ['asym'], 
                        'algorithm': ['RTN'], 
                    },
                },
            },
            recipes={
                'gptq_args':{'percdamp': 0.01},
                'awq_args':{'alpha': 'auto', 'clip': True},
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_name_dict={
                'fc1':{ 	# re.match
                    "weight": {
                        'bit': [4], # 1-8 bit 
                        'group_size': [32],  # 1 - 1024 or higher
                        'scheme': ['sym'], 
                        'algorithm': ['RTN'], 
                    },
                },
                'fc2':{ 	# re.match
                    "weight": {
                        'bit': [3], # 1-8 bit 
                        'group_size': [16],  # 1 - 1024 or higher
                        'scheme': ['asym'], 
                        'algorithm': ['RTN'], 
                    },
                },
                'fc3':{ 	# re.match
                    "weight": {
                        'dtype': ['fp32'],
                    },
                },
            },
            recipes={
                'gptq_args':{'percdamp': 0.01},
                'awq_args':{'alpha': 'auto', 'clip': True},
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        q_model.save('saved')


if __name__ == "__main__":
    unittest.main()
