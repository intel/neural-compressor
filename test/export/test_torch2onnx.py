import os
import copy
import shutil
import torch
import unittest
import numpy as np
import copy
from neural_compressor import quantization
from neural_compressor.experimental.common import Model
from neural_compressor.config import Torch2ONNXConfig
from neural_compressor import PostTrainingQuantConfig, QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression
from neural_compressor.data import Datasets, DATALOADERS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neural_compressor.utils.constant import FP32
import torch.utils.data as data


def train_func_cv(compression_manager, model):
    compression_manager.callbacks.on_train_begin()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    loss = output[0].mean() if isinstance(output, tuple) else output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    compression_manager.callbacks.on_train_end()
    return model

def train_func_nlp(compression_manager, model, input):
    compression_manager.callbacks.on_train_begin()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model.train()
    output = model(**input)
    loss = output.logits[0][0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    compression_manager.callbacks.on_train_end()
    return model

def check_CV_onnx(model_path, dataloader):
    import onnxruntime as ort
    ort_session = ort.InferenceSession(model_path)
    it = iter(dataloader)
    input = next(it)
    input_dict = {'input': input[0].detach().cpu().numpy()}
    ort_session.run(None, input_dict)
    return True

def check_NLP_onnx(model_path, input):
    import onnxruntime as ort
    ort_session = ort.InferenceSession(model_path, None)
    input_dict = {}
    for k, v in input.items():
        input_dict[k] = np.array(v)
    ort_session.run(None, input_dict)
    return True

# This fake eval_func is used to avoid performance_only setting,
# which will overwrite the fp32 model
def eval_func(model):
    return 1

class DummyNLPDataloader(object):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors='pt')
        self.encoded_dict['labels'] = 1
        self.batch_size = 1

    def __iter__(self):
        yield self.encoded_dict

    def __next__(self):
        return self.encoded_dict

class TestPytorch2ONNX(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from torchvision.models.quantization import resnet18
        self.cv_model = resnet18()
        self.cv_dataset = Datasets("pytorch")["dummy"]((10, 3, 224, 224))
        self.cv_dataloader = DATALOADERS["pytorch"](self.cv_dataset)
        self.nlp_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.nlp_dataloader = DummyNLPDataloader(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        input = next(self.nlp_dataloader)
        input.pop('labels')
        self.nlp_input = input

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('nc_workspace', ignore_errors=True)
        os.remove('fp32-cv-model.onnx')
        os.remove('int8-cv-qdq-model.onnx')
        os.remove('int8-cv-qlinear-model.onnx')
        os.remove('fp32-nlp-model.onnx')
        os.remove('int8-nlp-qdq-model.onnx')
        os.remove('int8-nlp-qlinear-model.onnx')

    def test_fp32_CV_models(self):
        model = copy.deepcopy(self.cv_model)
        inc_model = Model(model)
        fp32_onnx_config = Torch2ONNXConfig(
            dtype="fp32",
            example_inputs=torch.randn(1, 3, 224, 224),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={"input": {0: "batch_size"},
                            "output": {0: "batch_size"}},
        )
        inc_model.export('fp32-cv-model.onnx', fp32_onnx_config)
        check_CV_onnx('fp32-cv-model.onnx', self.cv_dataloader)

    def test_int8_CV_models(self):
        for fake_yaml in ["static", "qat", "dynamic"]:
            model = copy.deepcopy(self.cv_model)
            if fake_yaml == "qat":
                quant_conf = QuantizationAwareTrainingConfig()
                compression_manager = prepare_compression(copy.deepcopy(model), quant_conf)
                q_model = train_func_cv(compression_manager, compression_manager.model)
            else:
                if fake_yaml == "dynamic":
                    quant_conf = PostTrainingQuantConfig(approach="dynamic")
                elif fake_yaml == "static":
                    # Random fallback one op to test
                    fallback_op= {
                        "conv1": {
                            "activation": {"dtype": ["fp32"]},  
                            "weight": {"dtype": ["fp32"]}
                        }
                    }
                    quant_conf = PostTrainingQuantConfig(
                        approach="static",
                        op_name_dict=fallback_op,
                    )
                q_model = quantization.fit(
                    model,
                    quant_conf,
                    eval_func=eval_func,
                    calib_dataloader=self.cv_dataloader if fake_yaml == "static" else None)

            int8_onnx_config = Torch2ONNXConfig(
                dtype="int8",
                opset_version=14,
                quant_format="QDQ",
                example_inputs=torch.randn(1, 3, 224, 224),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={"input": {0: "batch_size"},
                              "output": {0: "batch_size"}},
            )
            q_model.export('int8-cv-qdq-model.onnx', int8_onnx_config)
            check_CV_onnx('int8-cv-qdq-model.onnx', self.cv_dataloader)

            int8_onnx_config = Torch2ONNXConfig(
                dtype="int8",
                opset_version=14,
                quant_format="QLinear",
                example_inputs=torch.randn(1, 3, 224, 224),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={"input": {0: "batch_size"},
                              "output": {0: "batch_size"}},
            )
            q_model.export('int8-cv-qlinear-model.onnx', int8_onnx_config)
            check_CV_onnx('int8-cv-qlinear-model.onnx', self.cv_dataloader)


    def test_fp32_NLP_models(self):
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        dynamic_axes = {k: symbolic_names for k in self.nlp_input.keys()}

        model = copy.deepcopy(self.nlp_model)
        inc_model = Model(model)
        fp32_onnx_config = Torch2ONNXConfig(
            dtype="fp32",
            example_inputs=self.nlp_input,
            input_names=list(self.nlp_input.keys()),
            output_names=['labels'],
            dynamic_axes=dynamic_axes,
        )
        inc_model.export('fp32-nlp-model.onnx', fp32_onnx_config)
        check_NLP_onnx('fp32-nlp-model.onnx', self.nlp_input)

    def test_int8_NLP_models(self):
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        dynamic_axes = {k: symbolic_names for k in self.nlp_input.keys()}

        for fake_yaml in ["static", "qat", "dynamic"]:
            model = copy.deepcopy(self.nlp_model)
            if fake_yaml == "qat":
                quant_conf = QuantizationAwareTrainingConfig(
                    op_type_dict={"Embedding":FP32},
                )
                compression_manager = prepare_compression(copy.deepcopy(model), quant_conf)
                q_model = train_func_nlp(
                    compression_manager,
                    compression_manager.model,
                    self.nlp_input
                )
            else:
                if fake_yaml == "dynamic":
                    quant_conf = PostTrainingQuantConfig(approach="dynamic")
                elif fake_yaml == "static":
                    # Random fallback one op to test
                    fallback_op= {
                        "distilbert.transformer.layer.5.ffn.lin2": {
                            "activation": {"dtype": ["fp32"]}, 
                            "weight": {"dtype": ["fp32"]}
                        }
                    }
                    quant_conf = PostTrainingQuantConfig(
                        approach="static",
                        op_name_dict=fallback_op,
                        op_type_dict={"Embedding":FP32},
                    )
                
                q_model = quantization.fit(
                    model,
                    quant_conf,
                    eval_func=eval_func,
                    calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None)

            int8_onnx_config = Torch2ONNXConfig(
                dtype="int8",
                opset_version=14,
                quant_format="QDQ",
                example_inputs=tuple(self.nlp_input.values()),
                input_names=list(self.nlp_input.keys()),
                output_names=['labels'],
                dynamic_axes=dynamic_axes,
            )
            q_model.export('int8-nlp-qdq-model.onnx', int8_onnx_config)
            check_NLP_onnx('int8-nlp-qdq-model.onnx', self.nlp_input)

            int8_onnx_config = Torch2ONNXConfig(
                dtype="int8",
                opset_version=14,
                quant_format="QLinear",
                example_inputs=tuple(self.nlp_input.values()),
                input_names=list(self.nlp_input.keys()),
                output_names=['labels'],
                dynamic_axes=dynamic_axes,
            )
            q_model.export('int8-nlp-qlinear-model.onnx', int8_onnx_config)
            check_NLP_onnx('int8-nlp-qlinear-model.onnx', self.nlp_input)

if __name__ == "__main__":
    unittest.main()

