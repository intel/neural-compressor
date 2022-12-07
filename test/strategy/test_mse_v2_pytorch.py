import os
import copy
import shutil
import unittest
import torchvision
from neural_compressor.experimental import Quantization, common


def build_mse_yaml():
    mse_yaml = '''
    model:
        name: resnet18
        framework: pytorch_fx

    tuning:
        strategy:
            name: mse_v2
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
    '''
    with open('mse_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(mse_yaml)


i=0
def eval_func(model):
    # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    # 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    acc_lst = [1, 1, 0, 0, 0, 0, 1, 1.1, 1.5, 1.1]
    
    global i
    i += 1
    if i == 1:
        return acc_lst[i]
    elif i <= 7:
        return acc_lst[i]
    elif 10 >= i > 7:
        return acc_lst[i]
    elif i > 10:
        return acc_lst[i]


class TestMSEV2Strategy_PyTorch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_mse_yaml()
        self.model = torchvision.models.resnet18()

    @classmethod
    def tearDownClass(self):
        os.remove('mse_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_quantization_saved(self):
        model = copy.deepcopy(self.model)
        quantizer = Quantization('mse_yaml.yaml')
        dataset = quantizer.dataset('dummy', (1, 3, 224, 224))
        quantizer.model = model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        q_model.save('./saved')

if __name__ == "__main__":
    unittest.main()
