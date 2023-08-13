import sys
import copy
sys.path.append("./")
import os
import shutil
import torch
import unittest
import transformers
from neural_compressor import quantization, PostTrainingQuantConfig

from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(30, 40)
        self.fc2 = torch.nn.Linear(40, 30)
        self.fc3 = torch.nn.Linear(30, 5)

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
        self.gptj.seqlen = 512
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
        compressed_model = q_model.export_compressed_model()
        out3 = compressed_model(input)
        self.assertTrue(torch.all(out3==out2))

        model = Model()
        out1 = model(input)

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            recipes={
                # By default, sym_full_range is False and 4 bit sym will only use range [-7,7].
                'rtn_args': {'sym_full_range': True}
            }
        )
        q_model = quantization.fit(model, conf)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))
        compressed_model = q_model.export_compressed_model(sym_full_range=True)
        out3 = compressed_model(input)
        self.assertTrue(torch.all(out3==out2))

        model = Model()
        out1 = model(input)
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

        model = Model()
        out1 = model(input)
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

        model = Model()
        out1 = model(input)
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


        model_size1 = os.path.getsize('saved/best_model.pt')/1024
        print("FP32 Model size:{:.3f}M".format(model_size1))
        from neural_compressor.model import Model as INCModel
        inc_model = INCModel(new_model)
        inc_model.export_compressed_model(qweight_config_path = 'saved/qconfig.json')
        torch.save(inc_model.state_dict(), 'saved/tmp.pt')
        model_size2 = os.path.getsize('saved/tmp.pt')/1024
        print("WeightOnlyLinear Model size:{:.3f}M".format(model_size2))
        self.assertTrue(isinstance(inc_model.model.fc1, WeightOnlyLinear))
        self.assertTrue(model_size1 / model_size2 > 2)

    def test_AWQ_quant(self):
        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bits': 4, # 1-8 bits 
                        'group_size': 32,  # -1 (per-channel)
                        'scheme': 'asym', 
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
                'awq_args':{'auto_scale': True, 'mse_range': True, 'folding': False},
            },
        )
        fp32_model = copy.deepcopy(self.gptj)
        q_model = quantization.fit(
            self.gptj, 
            conf, 
            calib_dataloader=self.llm_dataloader,
        )
        q_model.save('saved')
        input = torch.ones([1, 10], dtype=torch.long)
        out1 = q_model(input)
        from neural_compressor.utils.pytorch import load
        reload_model = load('saved', fp32_model, weight_only=True)
        out2 = reload_model(input)
        q_model.export_compressed_model()
        out3 = q_model(input)
        # no idea about the gap at 1e-08, use allclose instead of out1==out2
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))
        self.assertTrue(torch.allclose(out1[0], out3[0], atol=1e-05))
        self.assertTrue(isinstance(q_model.model.transformer.h[0].mlp.fc_in, WeightOnlyLinear))
        self.assertTrue(isinstance(q_model.model.lm_head, torch.nn.Linear))

    def test_GPTQ_quant(self):
        class gptq_inc_loader(object):
            def __init__(self, nsamples=32):
                self.batch_size = 1
                self.nsamples = nsamples
            
            def __len__(self):
                return self.nsamples // self.batch_size

            def __iter__(self):
                for i in range(self.nsamples):
                    yield (torch.ones([1, 512], dtype=torch.long), torch.ones([1, 512], dtype=torch.long))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{ 	# re.match
                    "weight": {
                        'bits': 4, # 1-8 bits 
                        'group_size': 8,  # -1 (per-channel)
                        'scheme': 'sym', 
                        'algorithm': 'GPTQ', 
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
                'gptq_args':{'percdamp': 0.01, 'actorder': False},
            },
        )
        input = (torch.ones([1, 512], dtype=torch.long))
        dataloader = gptq_inc_loader()
        q_model = quantization.fit(self.gptj, conf, calib_dataloader=dataloader,)
        q_model.save('saved')
        out1 = q_model.model(*input)
        compressed_model = q_model.export_compressed_model()
        out2 = compressed_model(*input)
        torch.save(compressed_model.state_dict(), 'saved/compressed_model.pt')
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))
        print("GPTQ Done")

    def test_TEQ_quant(self):
        class teq_inc_loader(object):
            def __init__(self, nsamples=32):
                self.batch_size = 1
                self.nsamples = nsamples

            def __len__(self):
                return self.nsamples // self.batch_size

            def __iter__(self):
                for i in range(self.nsamples):
                    yield (torch.ones([1, 512], dtype=torch.long), torch.ones([1, 512], dtype=torch.long))

        conf = PostTrainingQuantConfig(
            approach='weight_only',
            op_type_dict={
                '.*':{  # re.match
                    "weight": {
                        'bits': 4, # 1-8 bits
                        'group_size': 32,  # -1 (per-channel)
                        'scheme': 'sym',
                        'algorithm': 'TEQ',
                    },
                },
            },
            op_name_dict={
                '.*lm_head':{   # re.match
                    "weight": {
                        'dtype': 'fp32'
                    },
                },
            },
            recipes={
                'teq_args':{"folding": True},
            },
        )

        dataloader = teq_inc_loader()

        q_model = quantization.fit(self.gptj, conf, calib_dataloader=dataloader,)

if __name__ == "__main__":
    unittest.main()
