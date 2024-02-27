import os
import shutil
import unittest
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnx import TensorProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto
from packaging.version import Version
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.adaptor.pytorch import get_torch_version
from neural_compressor.conf.config import conf
from neural_compressor.data import DATALOADERS, DataLoader, Datasets
from neural_compressor.experimental import Benchmark, Quantization, common
from neural_compressor.model import Model


def build_static_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          calibration:
            sampling_size: 50
          op_wise: {
            'Gather_*': {
            'activation':  {'dtype': ['fp32'], 'scheme':['sym']},
            'weight': {'dtype': ['fp32'], 'scheme':['sym']}
            }
          }

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: ./nc_workspace/recover/
        """
    with open("qlinear.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qdq

        quantization:
          approach: post_training_static_quant
          calibration:
            sampling_size: 50
          op_wise: {
            'Gather_*': {
            'activation':  {'dtype': ['fp32'], 'scheme':['sym']},
            'weight': {'dtype': ['fp32'], 'scheme':['sym']}
            }
          }

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: ./nc_workspace/recover/
        """
    with open("qdq.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_benchmark_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        evaluation:
          performance:
            warmup: 1
            iteration: 10
            configs:
              num_of_instance: 1
            dataloader:
              batch_size: 1
              dataset:
                ImageFolder:
                  root: /path/to/evaluation/dataset/
          accuracy:
            metric:
              topk: 1

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("benchmark.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_dynamic_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_integerops

        quantization:
          approach: post_training_dynamic_quant
          calibration:
              sampling_size: 50

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: ./nc_workspace/recover/

        """
    with open("dynamic.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_recipe_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          recipes:
            first_conv_or_matmul_quantization: False
            last_conv_or_matmul_quantization: False
          calibration:
            sampling_size: 1
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        tuning:
          accuracy_criterion:
            relative:  -0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("recipe.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_recipe2_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          recipes:
            last_conv_or_matmul_quantization: False
            pre_post_process_quantization: False
          calibration:
            sampling_size: 1
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        tuning:
          accuracy_criterion:
            relative:  -0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("recipe2.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_gather_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          calibration:
            sampling_size: 1
            dataloader:
              batch_size: 1
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        evaluation:
          accuracy:
            metric:
              MSE:
                compare_label: False
            dataloader:
              batch_size: 1
              dataset:
                dummy_v2:
                  input_shape: [100, 4]

        tuning:
          accuracy_criterion:
            relative:  -0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("gather.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_rename_yaml():
    fake_yaml = """
        model:
          name: test
          framework: onnxrt_integerops

        quantization:
          approach: post_training_dynamic_quant
          calibration:
              sampling_size: 1

        evaluation:
          accuracy:
            metric:
              Accuracy: {}

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("rename.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def build_non_MSE_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          calibration:
              sampling_size: 50
          op_wise: {
            'Gather_*': {
            'activation':  {'dtype': ['fp32'], 'scheme':['sym']},
            'weight': {'dtype': ['fp32'], 'scheme':['sym']}
            }
          }

        evaluation:
          accuracy:
            metric:
              MSE:
               compare_label: False
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            relative:  0.1
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: ./nc_workspace/recover/

        """
    with open("non_MSE.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)


def eval_func(model):
    return 1.0


def export_onnx_cv_model(model, path, opset=12):
    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset,  # the ONNX version to export the model to, please ensure at least 11.
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model"s input names
        output_names=["output"],  # the model"s output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )


def export_onnx_nlp_model(model, path, opset=12):
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    inputs = {
        "input_ids": torch.ones(1, 128, dtype=torch.int64),
        "attention_mask": torch.ones(1, 128, dtype=torch.int64),
    }
    torch.onnx.export(
        model,  # model being run
        (inputs["input_ids"], inputs["attention_mask"]),  # model input (or a tuple for multiple inputs)
        path,  # where to save the model (can be a file or file-like object)
        opset_version=opset,  # the ONNX version to export the model
        do_constant_folding=True,  # whether to execute constant folding
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["logits"],
        dynamic_axes={"input_ids": symbolic_names, "attention_mask": symbolic_names},  # variable length axes
    )


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """Helper function to generate initializers for test inputs."""
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


def build_ir3_model():
    input0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 2048])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1000])
    weight = helper.make_tensor_value_info("X1_weight", TensorProto.FLOAT, [1000, 2048])

    X1_weight = generate_input_initializer([1000, 2048], np.float32, "X1_weight")
    kwargs = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}
    gemm = helper.make_node("Gemm", ["input0", "X1_weight"], ["output"], name="gemm", **kwargs)

    graph = helper.make_graph([gemm], "test_graph_6", [input0], [output])
    graph.initializer.add().CopyFrom(X1_weight)
    graph.input.extend([weight])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 11)]})
    model.ir_version = 3
    return model


def build_matmul_model():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 5, 2])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(10)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [B_init, E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_matmul_model2():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 5, 1])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 1, 5, 1])

    C1_init = helper.make_tensor("C1", TensorProto.FLOAT, [1, 1, 5, 5], np.random.random(25).tolist())
    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    matmul_node2 = onnx.helper.make_node("MatMul", ["C1", "C"], ["C2"], name="Matmul2")
    matmul_node3 = onnx.helper.make_node("MatMul", ["A", "C2"], ["C3"], name="Matmul3")
    e_value = np.random.randint(2, size=(5)).astype(np.float32)
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add = onnx.helper.make_node("Add", ["C3", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(5)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 1, 5, 1], e_value.reshape(5).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph(
        [matmul_node, matmul_node2, matmul_node3, add, add2], "test_graph_1", [A, B], [H], [E_init, F_init, C1_init]
    )
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_matmul_model3():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 5, 2])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="post_quant_Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(10)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [B_init, E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_matmul_gather_model():
    input = helper.make_tensor_value_info("input0", TensorProto.INT64, [1, 1])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1])

    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    squeeze = onnx.helper.make_node("Squeeze", ["input0", "axes"], ["A"], name="squeeze")

    b_value = np.random.random((1, 2048))
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [1, 2048], b_value.reshape(2048).tolist())

    gather = onnx.helper.make_node("Gather", ["B", "A"], ["C"], name="gather")

    d_value = np.random.random((2048, 1)).astype("float32")
    D_init = helper.make_tensor("D", TensorProto.FLOAT, [2048, 1], d_value.reshape(2048).tolist())
    matmul = onnx.helper.make_node("MatMul", ["C", "D"], ["output0"])

    graph = helper.make_graph([squeeze, gather, matmul], "test_graph_1", [input], [output], [B_init, D_init, axes])
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_model_with_gather():
    b_value = np.random.randint(2, size=(10)).astype(np.int32)
    B_init = helper.make_tensor("B", TensorProto.INT32, [10], b_value.reshape(10).tolist())
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 100, 4])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [100, 4])
    squeeze = onnx.helper.make_node("Squeeze", ["A"], ["D"], name="squeeze")
    B = helper.make_tensor_value_info("B", TensorProto.INT32, [10])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [10, 4])
    node = onnx.helper.make_node("Gather", ["D", "B"], ["C"], name="gather")
    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [10, 1], e_value.reshape(10).tolist())
    F = helper.make_tensor_value_info("F", TensorProto.FLOAT, [10, 4])
    add = onnx.helper.make_node("Add", ["C", "E"], ["F"], name="add")
    graph = helper.make_graph([squeeze, node, add], "test_graph_1", [A], [F], [B_init, E_init])
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_rename_model():
    input_shape = [1, 1, 200]
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    w_shape = [2, 400, 200]
    w_weights = np.random.random_sample(w_shape).astype(dtype="float32")
    w_init = onnx.numpy_helper.from_array(w_weights, name="w")
    r_shape = [2, 400, 100]
    r_weights = np.random.random_sample(r_shape).astype(dtype="float32")
    r_init = onnx.numpy_helper.from_array(r_weights, name="r")
    b_shape = [2, 800]
    b_weights = np.random.random_sample(b_shape).astype(dtype="float32")
    b_init = onnx.numpy_helper.from_array(b_weights, name="b")
    kwargs = {}
    kwargs["direction"] = "bidirectional"
    kwargs["activations"] = ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"]
    kwargs["hidden_size"] = 100
    kwargs["input_forget"] = 0
    lstm_node = helper.make_node("LSTM", ["input", "w", "r", "b"], ["out"], name="lstm", **kwargs)

    b_value = np.random.randint(2, size=(1)).astype(np.int32)
    B_init = helper.make_tensor("B", TensorProto.INT32, [1], b_value.reshape(1).tolist())
    squeeze = onnx.helper.make_node("Squeeze", ["out"], ["D"], name="")
    B = helper.make_tensor_value_info("B", TensorProto.INT32, [1])
    node = onnx.helper.make_node("Gather", ["D", "B"], ["C"], name="")
    e_value = np.random.randint(2, size=(100)).astype(np.float32)
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 1, 100], e_value.reshape(100).tolist())
    F = helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 1, 100])
    add = onnx.helper.make_node("Add", ["C", "E"], ["F"], name="")
    graph = helper.make_graph(
        [lstm_node, squeeze, node, add], "test_graph_1", [input_tensor], [F], [B_init, E_init, w_init, r_init, b_init]
    )
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_conv_model():
    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    conv1_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name="conv1_weight"
    )
    conv1_node = helper.make_node("Conv", ["input", "conv1_weight"], ["conv1_output"], name="conv1")

    conv2_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name="conv2_weight"
    )
    conv2_node = helper.make_node("Conv", ["conv1_output", "conv2_weight"], ["conv2_output"], name="conv2")

    conv3_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name="conv3_weight"
    )
    conv3_node = helper.make_node("Conv", ["input", "conv3_weight"], ["conv3_output"], name="conv3")

    avg_args = {"kernel_shape": [3, 3]}
    avgpool_node = helper.make_node("AveragePool", ["conv3_output"], ["avg_output"], name="AveragePool", **avg_args)

    concat_node = helper.make_node("Concat", ["avg_output", "conv2_output"], ["concat_output"], name="Concat", axis=1)
    output = helper.make_tensor_value_info("concat_output", TensorProto.FLOAT, [1, 8, 220, 220])
    initializers = [conv1_weight_initializer, conv2_weight_initializer, conv3_weight_initializer]
    graph = helper.make_graph(
        [conv1_node, conv2_node, conv3_node, concat_node, avgpool_node],
        "test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def build_conv_model2():
    input0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 3, 1, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 3])

    X1_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X1_weight")
    X1_bias = generate_input_initializer([3], np.float32, "X1_bias")
    X3_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X3_weight")
    X3_bias = generate_input_initializer([3], np.float32, "X3_bias")
    X5_weight = generate_input_initializer([3, 3, 1, 1], np.float32, "X5_weight")
    X5_bias = generate_input_initializer([3], np.float32, "X5_bias")

    relu_node_1 = onnx.helper.make_node("Relu", ["input0"], ["X1"], name="Relu1")
    conv_node_1 = onnx.helper.make_node("Conv", ["X1", "X1_weight", "X1_bias"], ["X2"], name="Conv1")
    relu_node_2 = onnx.helper.make_node("Relu", ["X2"], ["X3"], name="Relu2")
    conv_node_2 = onnx.helper.make_node("Conv", ["X3", "X3_weight", "X3_bias"], ["X4"], name="Conv2")
    conv_node_3 = onnx.helper.make_node("Conv", ["X1", "X5_weight", "X5_bias"], ["X5"], name="Conv3")
    add_node = onnx.helper.make_node("Add", ["X4", "X5"], ["output"], name="Add")

    graph = helper.make_graph(
        [relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node], "test_graph_1", [input0], [output]
    )
    graph.initializer.add().CopyFrom(X1_weight)
    graph.initializer.add().CopyFrom(X1_bias)
    graph.initializer.add().CopyFrom(X3_weight)
    graph.initializer.add().CopyFrom(X3_bias)
    graph.initializer.add().CopyFrom(X5_weight)
    graph.initializer.add().CopyFrom(X5_bias)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_conv_model3():
    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    conv1_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name="conv1_weight"
    )
    conv1_node = helper.make_node("Conv", ["input", "conv1_weight"], ["conv1_output"], name="conv1")

    conv2_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name="conv2_weight"
    )
    conv2_node = helper.make_node("Conv", ["conv1_output", "conv2_weight"], ["conv2_output"], name="pre_quant_conv2")

    conv3_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name="conv3_weight"
    )
    conv3_node = helper.make_node("Conv", ["input", "conv3_weight"], ["conv3_output"], name="conv3")

    avg_args = {"kernel_shape": [3, 3]}
    avgpool_node = helper.make_node("AveragePool", ["conv3_output"], ["avg_output"], name="AveragePool", **avg_args)

    concat_node = helper.make_node("Concat", ["avg_output", "conv2_output"], ["concat_output"], name="Concat", axis=1)
    output = helper.make_tensor_value_info("concat_output", TensorProto.FLOAT, [1, 8, 220, 220])
    initializers = [conv1_weight_initializer, conv2_weight_initializer, conv3_weight_initializer]
    graph = helper.make_graph(
        [conv1_node, conv2_node, conv3_node, concat_node, avgpool_node],
        "test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def build_gemm_model():
    initializers = []
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [-1, 2048])
    weight_data = np.random.normal(0, 0.1, [10, 2048]).astype(np.float32)
    initializers.append(onnx.numpy_helper.from_array(weight_data, name="weight"))
    bias_data = np.random.normal(0, 0.1, [10]).astype(np.float32)
    initializers.append(onnx.numpy_helper.from_array(bias_data, name="bias"))
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [-1, 10])
    gemm = onnx.helper.make_node(
        "Gemm", ["input", "weight", "bias"], ["output"], alpha=1.0, beta=1.0, transB=1, name="gemm"
    )

    graph = helper.make_graph([gemm], "test", [input_tensor], [output_tensor], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def build_model_share_init():
    initializers = []
    input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    conv_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [222, 3, 3, 3]).astype(np.float32), name="conv_weight"
    )
    conv_bias_initializer = numpy_helper.from_array(np.random.randint(1, 2, [222]).astype(np.float32), name="conv_bias")
    conv_node = helper.make_node("Conv", ["input", "conv_weight", "conv_bias"], ["conv_output"], name="conv")

    add_node = helper.make_node("Add", ["conv_bias", "conv_output"], ["add_output"], name="add")

    div_node = helper.make_node("Div", ["add_output", "conv_bias"], ["div_output"], name="div")

    output = helper.make_tensor_value_info("div_output", TensorProto.FLOAT, [1, 222, 222, 222])
    initializers = [conv_weight_initializer, conv_bias_initializer]
    graph = helper.make_graph(
        [conv_node, add_node, div_node],
        "test",
        [input],
        [output],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def build_benchmark():
    seq = """
from neural_compressor.experimental import Benchmark
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor import conf
from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, numpy_helper
from onnxruntime_extensions import onnx_op
import numpy as np

@onnx_op(op_type="PyReverseMatrix")
def reverse_matrix(x):
    # The user custom op implementation here.
    return np.flip(x, axis=0).astype(np.float32)

nodes = []
nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
nodes[1:] = [helper.make_node('PyReverseMatrix',
                              ['identity1'], ['reversed'],
                              domain='ai.onnx.contrib')]

input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
output0 = helper.make_tensor_value_info(
        'reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

graph = helper.make_graph(nodes, 'test0', [input0], [output0])
model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})

datasets = Datasets('onnxrt_qlinearops')
ext_dataset = datasets['dummy'](shape=(10, 2), low=0., high=1., label=True)
ext_dataloader = DATALOADERS['onnxrt_qlinearops'](ext_dataset)

conf.model.framework = 'onnxrt_qlinearops'
conf.evaluation.accuracy.metric = {'Accuracy': {}}
evaluator = Benchmark(conf)
evaluator.b_dataloader = ext_dataloader
evaluator.model = model
evaluator('performance')
    """
    with open("benchmark.py", "w", encoding="utf-8") as f:
        f.writelines(seq)


class MatmulDataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(3):
            self.data.append(np.random.randn(5, 5).astype("float32"))
            self.label.append(np.random.randn(5, 1).astype("float32"))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class DummyNLPDataloader(object):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt")
        self.encoded_dict["labels"] = 1
        self.batch_size = 1


class DummyNLPDataloader_list(DummyNLPDataloader):
    def __init__(self, model_name):
        super().__init__(model_name)

    def __iter__(self):
        yield [self.encoded_dict["input_ids"], self.encoded_dict["attention_mask"]], self.encoded_dict["labels"]


class DummyNLPDataloader_dict(DummyNLPDataloader):
    def __init__(self, model_name):
        super().__init__(model_name)

    def __iter__(self):
        yield {k: v.numpy().tolist() for k, v in self.encoded_dict.items() if k != "labels"}, self.encoded_dict[
            "labels"
        ]


class DummyCVDataset(object):
    def __init__(self, shape):
        np.random.seed(9527)
        self.label = True
        self.shape = [shape]
        self.low = [0.0]
        self.high = [1.0]
        self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.label:
            return sample, 0
        else:
            return sample


class DummyCVDataset_list(DummyCVDataset):
    def __init__(self, shape):
        super().__init__(shape)
        self.process()

    def process(self):
        for idx in range(0, len(self.shape)):
            tensor = np.random.uniform(low=self.low[idx], high=self.high[idx], size=self.shape[idx])
            tensor = tensor.astype(np.float32)
            self.dataset.append(tensor)


class DummyCVDataset_dict(DummyCVDataset):
    def __init__(self, shape):
        super().__init__(shape)
        self.process()

    def process(self):
        for idx in range(0, len(self.shape)):
            tensor = np.random.uniform(low=self.low[idx], high=self.high[idx], size=self.shape[idx])
            tensor = tensor.astype(np.float32)
            self.dataset.append({"input": tensor})


class TestAdaptorONNXRT(unittest.TestCase):
    mb_v2_export_path = "mb_v2.onnx"
    mb_v2_model = torchvision.models.mobilenet_v2()
    rn50_export_path = "rn50.onnx"
    rn50_model = torchvision.models.resnet50()

    model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=AutoConfig.from_pretrained(model_name_or_path)
    )
    distilbert_export_path = "distilbert.onnx"

    model_name_or_path = "Alireza1044/albert-base-v2-sst2"
    albert_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=AutoConfig.from_pretrained(model_name_or_path)
    )
    albert_export_path = "albert.onnx"

    datasets = Datasets("onnxrt_qlinearops")
    cv_dataset = datasets["dummy"](shape=(10, 3, 224, 224), low=0.0, high=1.0, label=True)
    cv_dataloader = DATALOADERS["onnxrt_qlinearops"](cv_dataset)

    ir3_dataset = datasets["dummy"](shape=(10, 2048), low=0.0, high=1.0, label=True)
    ir3_dataloader = DATALOADERS["onnxrt_qlinearops"](ir3_dataset)

    gather_dataset = Datasets("onnxrt_qlinearops")["dummy"](shape=(5, 100, 4), label=True)
    gather_dataloader = DATALOADERS["onnxrt_qlinearops"](gather_dataset)

    ext_dataset = datasets["dummy"](shape=(10, 2), low=0.0, high=1.0, label=True)
    ext_dataloader = DATALOADERS["onnxrt_qlinearops"](ext_dataset)

    rename_dataset = Datasets("onnxrt_qlinearops")["dummy"](shape=(5, 1, 200), label=True)
    rename_dataloader = DATALOADERS["onnxrt_qlinearops"](rename_dataset)

    matmul_dataset = MatmulDataset()
    matmul_dataloader = DATALOADERS["onnxrt_qlinearops"](matmul_dataset)

    conv_dataset = Datasets("onnxrt_qlinearops")["dummy"](shape=(10, 3, 1, 3), label=True)
    conv_dataloader = DATALOADERS["onnxrt_qlinearops"](conv_dataset)

    @classmethod
    def setUpClass(self):
        build_rename_yaml()
        build_static_yaml()
        build_dynamic_yaml()
        build_gather_yaml()
        build_non_MSE_yaml()
        build_benchmark_yaml()
        build_recipe_yaml()
        build_recipe2_yaml()
        export_onnx_cv_model(self.mb_v2_model, self.mb_v2_export_path, 13)
        self.mb_v2_model = onnx.load(self.mb_v2_export_path)
        export_onnx_cv_model(self.rn50_model, self.rn50_export_path, 12)
        export_onnx_cv_model(self.rn50_model, "rn50_9.onnx", 9)
        self.rn50_model = onnx.load(self.rn50_export_path)
        self.ir3_model = build_ir3_model()
        self.gather_model = build_model_with_gather()
        self.matmul_model = build_matmul_model()
        self.matmul_model2 = build_matmul_model2()
        self.matmul_model3 = build_matmul_model3()
        self.rename_model = build_rename_model()
        self.conv_model = build_conv_model()
        self.gemm_model = build_gemm_model()
        self.conv_model2 = build_conv_model2()
        self.conv_model3 = build_conv_model3()
        self.shared_init_model = build_model_share_init()
        export_onnx_nlp_model(self.distilbert_model, self.distilbert_export_path, 14)
        export_onnx_nlp_model(self.albert_model, self.albert_export_path, 14)
        self.distilbert_model = onnx.load(self.distilbert_export_path)
        self.albert_model = onnx.load(self.albert_export_path)
        self.gather_matmul_model = build_matmul_gather_model()
        build_benchmark()

    @classmethod
    def tearDownClass(self):
        os.remove("qlinear.yaml")
        os.remove("qdq.yaml")
        os.remove("recipe.yaml")
        os.remove("recipe2.yaml")
        os.remove("dynamic.yaml")
        os.remove("non_MSE.yaml")
        os.remove("benchmark.yaml")
        os.remove("gather.yaml")
        os.remove("rename.yaml")
        os.remove("rename_model.onnx")
        os.remove("rn50_9.onnx")
        os.remove(self.mb_v2_export_path)
        os.remove(self.rn50_export_path)
        os.remove("best_model.onnx")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_ext_model(self):
        import sys

        if sys.version_info < (3, 10):
            os.system("python benchmark.py")

    def test_adaptor_register(self):
        from neural_compressor.adaptor.adaptor import adaptor_registry

        def test():
            @adaptor_registry
            class ONNXRT_QLinearOpsAdaptor:
                def quantize(self):
                    pass

                def evaluate(self):
                    pass

        with self.assertRaises(ValueError):
            test()

    @unittest.skipIf(
        Version(ort.__version__) == Version("1.13.1"),
        "This function does not work with ONNX Runtime 1.13.1 for QDQ format quantization of ONNX models.",
    )
    def test_inspect_tensor(self):
        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "imagenet"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        op_list = [i.name for i in self.rn50_model.graph.node if i.op_type == "Conv"]

        data = adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type="activation", op_list=op_list)
        self.assertNotEqual(len(data["activation"][0]), 0)
        self.assertTrue("weight" not in data)

        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type="activation", save_to_disk=True)
        self.assertTrue(os.path.isfile(framework_specific_info["workspace_path"] + "inspect_result.pkl"))

        data = adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type="weight", op_list=op_list)
        self.assertNotEqual(len(data["weight"]), 0)
        self.assertTrue("activation" not in data)

        data = adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type="all", op_list=op_list)
        self.assertTrue("activation" in data)
        self.assertTrue("weight" in data)
        self.assertNotEqual(len(data["activation"][0]), 0)
        self.assertNotEqual(len(data["weight"]), 0)

        data = adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, [op_list[0]], inspect_type="activation")
        self.assertEqual(len(data["activation"][0]), 1)

        op = OrderedDict()
        op[(op_list[0], "Conv")] = None
        data = adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, op.keys(), inspect_type="activation")
        self.assertEqual(len(data["activation"][0]), 1)

        for fake_yaml in ["qlinear.yaml", "qdq.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = self.rn50_model
            q_model = quantizer.fit()
            self.assertNotEqual(q_model, None)

            adaptor._pre_optimize(common.Model(self.rn50_model))
            opt_model = quantizer.strategy.adaptor.pre_optimized_model

            op_list, _ = quantizer.strategy.adaptor.diagnosis_helper(
                opt_model, q_model, None, "./nc_workspace/recover/"
            )

            fp32_tensor = quantizer.strategy.adaptor.inspect_tensor(opt_model.model, self.cv_dataloader, op_list)
            int8_tensor = quantizer.strategy.adaptor.inspect_tensor(q_model.model, self.cv_dataloader, op_list)

            self.assertTrue(len(fp32_tensor["activation"]) == len(int8_tensor["activation"]))
            self.assertTrue(sorted(fp32_tensor["activation"][0].keys()) == sorted(int8_tensor["activation"][0].keys()))
            for op in op_list:
                for x, y in zip(fp32_tensor["activation"][0][op].values(), int8_tensor["activation"][0][op].values()):
                    self.assertTrue(x.shape == y.shape)

            if fake_yaml == "qlinear.yaml":
                fp32_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    opt_model.model, self.cv_dataloader, op_list, inspect_type="weight"
                )
                int8_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    q_model.model, self.cv_dataloader, op_list, inspect_type="weight"
                )
                self.assertTrue(len(fp32_tensor["weight"]) == len(int8_tensor["weight"]))
                self.assertTrue(sorted(fp32_tensor["weight"].keys()) == sorted(int8_tensor["weight"].keys()))
                ai_onnx_domain = [
                    opset for opset in q_model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
                ]
                if ai_onnx_domain[0].version > 12 or Version(ort.__version__) < Version("1.12.0"):
                    for op in fp32_tensor["weight"].keys():
                        self.assertTrue(
                            sorted(fp32_tensor["weight"][op].keys()) == sorted(int8_tensor["weight"][op].keys())
                        )
                fp32_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    opt_model.model, self.cv_dataloader, op_list, inspect_type="all"
                )
                int8_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    q_model.model, self.cv_dataloader, op_list, inspect_type="all"
                )
                self.assertTrue(len(fp32_tensor["weight"]) == len(int8_tensor["weight"]))
                self.assertTrue(len(fp32_tensor["activation"]) == len(int8_tensor["activation"]))
                self.assertTrue(sorted(fp32_tensor["weight"].keys()) == sorted(int8_tensor["weight"].keys()))
                if ai_onnx_domain[0].version > 12 or Version(ort.__version__) < Version("1.12.0"):
                    for op in fp32_tensor["weight"].keys():
                        self.assertTrue(
                            sorted(fp32_tensor["weight"][op].keys()) == sorted(int8_tensor["weight"][op].keys())
                        )
                self.assertTrue(
                    sorted(fp32_tensor["activation"][0].keys()) == sorted(int8_tensor["activation"][0].keys())
                )
                if ai_onnx_domain[0].version > 12 or Version(ort.__version__) < Version("1.12.0"):
                    for op in op_list:
                        self.assertTrue(
                            sorted(fp32_tensor["activation"][0][op].keys())
                            == sorted(int8_tensor["activation"][0][op].keys())
                        )

                config = PostTrainingQuantConfig(approach="static", recipes={"gemm_to_matmul": False})
                q_model = quantization.fit(self.gemm_model, config, calib_dataloader=self.ir3_dataloader)

                fp32_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    self.gemm_model, self.ir3_dataloader, ["gemm"], inspect_type="weight"
                )
                int8_tensor = quantizer.strategy.adaptor.inspect_tensor(
                    q_model.model, self.ir3_dataloader, ["gemm"], inspect_type="weight"
                )
                self.assertTrue(len(fp32_tensor["weight"]) == len(int8_tensor["weight"]))
                self.assertTrue(sorted(fp32_tensor["weight"].keys()) == sorted(int8_tensor["weight"].keys()))

    def test_set_tensor(self):
        from neural_compressor.adaptor.ox_utils.util import get_node_original_name, quantize_data_with_scale_zero

        config = PostTrainingQuantConfig(
            approach="static", recipes={"gemm_to_matmul": False, "graph_optimization_level": "ENABLE_EXTENDED"}
        )
        q_model = quantization.fit(self.mb_v2_model, config, calib_dataloader=self.cv_dataloader)

        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "imagenet"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        q_config = {
            get_node_original_name(q_model.nodes()[1]): {
                "weight": {"granularity": "per_channel", "dtype": onnx_proto.TensorProto.INT8, "scheme": "sym"}
            }
        }
        adaptor.quantize_config = q_config
        version = get_torch_version()
        q_model.save("./best_model.onnx")
        ai_onnx_domain = [
            opset for opset in q_model.model.opset_import if not opset.domain or opset.domain == "ai.onnx"
        ]
        weight_data = np.random.random([32, 3, 3, 3])
        bias_data = np.random.random([32])
        load_model = Model(onnx.load("best_model.onnx"))
        if version >= Version("1.7.0-rc1"):
            tensor_name = self.mb_v2_model.graph.node[0].input[1] + "_quantized"
            scale_tensor, zo_tensor = load_model.get_scale_zero(tensor_name)
            new_tensor = quantize_data_with_scale_zero(
                weight_data,
                onnx_proto.TensorProto.INT8,
                "sym",
                np.expand_dims(numpy_helper.to_array(scale_tensor), axis=(1, 2, 3)),
                np.expand_dims(numpy_helper.to_array(zo_tensor), axis=(1, 2, 3)),
            )
            adaptor.set_tensor(
                load_model.model,
                {tensor_name: weight_data},
            )
            self.assertTrue((new_tensor == numpy_helper.to_array(load_model.get_initializer(tensor_name))).all())

            tensor_name = self.mb_v2_model.graph.node[0].input[2]
            node = q_model.input_name_to_nodes[tensor_name + "_quantized"][0]
            input_scale = numpy_helper.to_array(q_model.get_initializer(node.input[1]))
            weight_scale = numpy_helper.to_array(q_model.get_initializer(node.input[4]))
            new_tensor = (bias_data / (input_scale * weight_scale)).round().astype(np.int32)
            adaptor.set_tensor(
                q_model,
                {tensor_name: bias_data},
            )
            self.assertTrue(
                (new_tensor == numpy_helper.to_array(q_model.get_initializer(tensor_name + "_quantized"))).all()
            )
        else:
            tensor_name = "ConvBnFusion_W_features.0.0.weight" + "_quantized"
            scale_tensor, zo_tensor = load_model.get_scale_zero(tensor_name)
            new_tensor = quantize_data_with_scale_zero(
                weight_data,
                onnx_proto.TensorProto.INT8,
                "sym",
                np.expand_dims(numpy_helper.to_array(scale_tensor), axis=(1, 2, 3)),
                np.expand_dims(numpy_helper.to_array(zo_tensor), axis=(1, 2, 3)),
            )
            adaptor.set_tensor(
                load_model.model,
                {tensor_name: weight_data},
            )
            self.assertTrue((new_tensor == numpy_helper.to_array(load_model.get_initializer(tensor_name))).all())

            tensor_name = "ConvBnFusion_BN_B_features.0.1.bias"
            node = q_model.input_name_to_nodes[tensor_name + "_quantized"][0]
            input_scale = numpy_helper.to_array(q_model.get_initializer(node.input[1]))
            weight_scale = numpy_helper.to_array(q_model.get_initializer(node.input[4]))
            new_tensor = (bias_data / (input_scale * weight_scale)).round().astype(np.int32)
            adaptor.set_tensor(
                q_model,
                {tensor_name: bias_data},
            )
            self.assertTrue(
                (new_tensor == numpy_helper.to_array(q_model.get_initializer(tensor_name + "_quantized"))).all()
            )

    def test_auto_quant(self):
        conf.model.framework = "onnxrt_qlinearops"
        conf.quantization.approach = "post_training_auto_quant"
        conf.quantization.optype_wise = {
            "Add|MatMul|Conv": {"weight": {"algorithm": ["minmax"]}, "activation": {"algorithm": ["minmax"]}}
        }
        conf.quantization.calibration.sampling_size = 1
        conf.tuning.exit_policy.timeout = 1000000
        conf.tuning.exit_policy.max_trials = 8
        conf.evaluation.accuracy.metric = {"MSE": {"compare_label": False}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        conf.model.framework = "onnxrt_qdq"
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    def test_auto_quant_v2(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.quantization import fit

        tuning_criterion = TuningCriterion(max_trials=8, timeout=10000)
        accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
        conf = PostTrainingQuantConfig(
            quant_level=1,
            approach="auto",
            op_type_dict={
                "Add|MatMul|Conv": {"weight": {"algorithm": ["minmax"]}, "activation": {"algorithm": ["minmax"]}}
            },
            tuning_criterion=tuning_criterion,
            accuracy_criterion=accuracy_criterion,
        )
        conf.framework = "onnxrt_qlinearops"
        q_model = fit(model=self.rn50_model, conf=conf, calib_dataloader=self.cv_dataloader, eval_func=lambda model: 1)
        self.assertIsNotNone(q_model)

    def test_quantize_data_per_channel(self):
        from neural_compressor.adaptor.ox_utils.util import quantize_data_per_channel

        tensor_value = np.ones([2, 1])
        qType = onnx_proto.TensorProto.INT8
        scale_value = np.array([1, 1])
        zo_value = np.array([0, 0])
        new_tensor_value = quantize_data_per_channel(tensor_value, 1, 254, qType, "sym")
        self.assertEqual(tensor_value.all(), new_tensor_value[-1].all())

    def test_adaptor(self):
        from neural_compressor.utils.constant import FP32, INT8_SYM_MINMAX_PERTENSOR, UINT8_ASYM_MINMAX_PERTENSOR

        # check op_wise has higher priority than optype_wise
        conf.model.framework = "onnxrt_qlinearops"
        conf.quantization.approach = "post_training_static_quant"
        conf.quantization.calibration.sampling_size = 1
        conf.quantization.optype_wise = {"Add": FP32}
        conf.quantization.op_wise = {
            "add": {"weight": INT8_SYM_MINMAX_PERTENSOR, "activation": UINT8_ASYM_MINMAX_PERTENSOR}
        }
        conf.evaluation.accuracy.metric = {"MSE": {"compare_label": False}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.matmul_dataloader
        quantizer.eval_dataloader = self.matmul_dataloader
        quantizer.model = self.matmul_model
        q_model = quantizer.fit()
        self.assertTrue("add2" in [i.name for i in q_model.nodes()])
        self.assertTrue("add_quant" in [i.name for i in q_model.nodes()])

        # check optype_wise has higher priority than model_wise
        conf.quantization.pop("op_wise")
        conf.quantization.model_wise = {"weight": INT8_SYM_MINMAX_PERTENSOR}
        conf.quantization.optype_wise = {"MatMul": {"weight": {"granularity": ["per_channel"]}}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.matmul_dataloader
        quantizer.eval_dataloader = self.matmul_dataloader
        quantizer.model = self.matmul_model
        q_model = quantizer.fit()
        self.assertEqual(len([i for i in q_model.initializer() if i.name == "B_scale"][0].float_data), 2)

        conf.quantization.pop("optype_wise")
        conf.quantization.pop("model_wise")

        # check rename renamed nodes function
        conf.model.framework = "onnxrt_integerops"
        conf.quantization.approach = "post_training_dynamic_quant"
        conf.quantization.calibration.sampling_size = 1
        conf.evaluation.accuracy.metric = {"MSE": {"compare_label": False}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.rename_dataloader
        quantizer.eval_dataloader = self.rename_dataloader
        quantizer.model = self.rename_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)
        self.assertEqual(len([i.name for i in q_model.nodes()]), len(set([i.name for i in q_model.nodes()])))

        # check large model quantization
        conf.model.framework = "onnxrt_integerops"
        conf.quantization.approach = "post_training_dynamic_quant"
        conf.quantization.calibration.sampling_size = 1
        conf.evaluation.accuracy.metric = {"MSE": {"compare_label": False}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.rename_dataloader
        quantizer.eval_dataloader = self.rename_dataloader
        onnx.save(self.rename_model, "rename_model.onnx")
        quantizer.model = "rename_model.onnx"
        # force set the model to large model
        quantizer.model._is_large_model = True
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        quantizer = Quantization("dynamic.yaml")
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        import copy

        # check opset version
        tmp_model = copy.deepcopy(self.rn50_model)
        tmp_model.opset_import[0].version = 10
        quantizer.model = tmp_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)
        tmp_model.opset_import.extend([onnx.helper.make_opsetid("", 11)])
        quantizer.model = tmp_model
        q_model = quantizer.fit()
        self.assertEqual(q_model, None)
        model = onnx.load("rn50_9.onnx")
        quantizer.model = model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        # check query quantizable_ops function
        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "imagenet"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        tune_cfg = {
            "calib_iteration": 1,
            "op": {
                ("gather", "Gather"): {
                    "activation": {"dtype": ["uint8"], "quant_mode": "static"},
                    "weight": {"dtype": ["uint8"]},
                },
                ("add", "Add"): {
                    "activation": {"dtype": ["uint8"], "quant_mode": "static"},
                    "weight": {"dtype": ["int8"]},
                },
                ("squeeze", "Squeeze"): {
                    "activation": {"dtype": ["uint8"], "quant_mode": "static"},
                    "weight": {"dtype": ["int8"]},
                },
            },
        }
        adaptor.quantize(tune_cfg, common.Model(self.gather_model), self.gather_dataloader)
        self.assertTrue(len(adaptor.quantizable_ops), 2)

        # check int8 + fp16 function
        framework_specific_info["device"] = "gpu"
        framework_specific_info["backend"] = "onnxrt_cuda_ep"

        tune_cfg = {
            "calib_iteration": 1,
            "op": {
                ("Matmul", "MatMul"): {
                    "activation": {"dtype": ["uint8"], "quant_mode": "static"},
                    "weight": {"dtype": ["int8"]},
                },
                ("add", "Add"): {"activation": {"dtype": "fp16", "quant_mode": "static"}, "weight": {"dtype": "fp16"}},
                ("add2", "Add"): {"activation": {"dtype": "fp16", "quant_mode": "static"}, "weight": {"dtype": "fp16"}},
            },
        }
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        model = adaptor.quantize(tune_cfg, common.Model(self.matmul_model), self.matmul_dataloader)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == "Cast"]), 2)

        for fake_yaml in ["gather.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.model = self.gather_model
            q_model = quantizer.fit()
            self.assertNotEqual(q_model, None)

            quantizer.model = self.matmul_model2
            q_model = quantizer.fit()  # error input shape test
            self.assertEqual(q_model, None)

            quantizer.eval_dataloader = self.matmul_dataloader
            q_model = quantizer.fit()  # error input shape test
            self.assertEqual(q_model, None)

            quantizer.calib_dataloader = self.matmul_dataloader
            quantizer.eval_dataloader = self.matmul_dataloader
            quantizer.model = self.matmul_model
            q_model = quantizer.fit()
            self.assertNotEqual(q_model, None)

        quantizer = Quantization("recipe.yaml")
        quantizer.model = self.matmul_model
        quantizer.calib_dataloader = self.matmul_dataloader
        quantizer.eval_dataloader = self.matmul_dataloader
        q_model = quantizer.fit()
        self.assertTrue("Matmul" in [i.name for i in q_model.nodes()])

        quantizer = Quantization("recipe2.yaml")
        quantizer.model = self.conv_model2
        quantizer.calib_dataloader = self.conv_dataloader
        quantizer.eval_dataloader = self.conv_dataloader
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        for fake_yaml in ["non_MSE.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = self.mb_v2_model
            q_model = quantizer.fit()
            self.assertNotEqual(q_model, None)

            # check recover model function
            from neural_compressor.utils.utility import recover

            model = recover(self.mb_v2_model, "./nc_workspace/recover/history.snapshot", 0)
            self.assertTrue(model.model == q_model.model)

    def test_qdq_settings(self):
        config = PostTrainingQuantConfig(
            approach="static", quant_format="QDQ", recipes={"add_qdq_pair_to_weight": True}
        )
        q_model = quantization.fit(self.ir3_model, config, calib_dataloader=self.ir3_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QuantizeLinear"]), 3)

        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QuantizeLinear"]), 3)

        config = PostTrainingQuantConfig(approach="static", quant_format="QDQ", recipes={"dedicated_qdq_pair": True})
        q_model = quantization.fit(self.conv_model, config, calib_dataloader=self.cv_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QuantizeLinear"]), 6)

        config = PostTrainingQuantConfig(
            approach="static", quant_format="QDQ", recipes={"optypes_to_exclude_output_quant": ["Conv"]}
        )
        q_model = quantization.fit(self.rn50_model, config, calib_dataloader=self.cv_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QuantizeLinear"]), 53)

    def test_model_name_checking(self):
        # some nodes have names that include `_quant`
        # static
        config = PostTrainingQuantConfig(approach="static", quant_format="QDQ", recipes={"dedicated_qdq_pair": True})
        q_model = quantization.fit(self.conv_model3, config, calib_dataloader=self.cv_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QuantizeLinear"]), 6)
        # dynamic
        config = PostTrainingQuantConfig(approach="dynamic")
        q_model = quantization.fit(self.matmul_model3, config, calib_dataloader=self.matmul_dataloader)
        self.assertTrue("MatMulInteger" in [i.op_type for i in q_model.nodes()])

    def test_lower_is_better_case(self):
        import time

        conf.model.framework = "onnxrt_qlinearops"
        conf.quantization.approach = "post_training_static_quant"
        conf.quantization.model_wise = {
            "weight": {"granularity": ["per_tensor"]},
            "activation": {"granularity": ["per_tensor"]},
        }
        conf.tuning.exit_policy.max_trials = 5
        conf.tuning.accuracy_criterion.relative = 0.01
        conf.tuning.accuracy_criterion.higher_is_better = False
        conf.tuning.exit_policy.timeout = 100

        result = [0.0, 0.1, 0.1005, 0.102, 0.1002, 0.102, 0.102]

        def sub_eval(model, result):
            time.sleep(0.001 * len(result))
            del result[0]
            return result[0]

        def eval(model):
            return sub_eval(model, result)

        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.model = self.matmul_model
        quantizer.calib_dataloader = self.matmul_dataloader
        quantizer.eval_func = eval
        q_model = quantizer.fit()
        node_names = [i.name for i in q_model.nodes()]
        # This assert it depends on the number of trials, disables it first.
        # self.assertTrue('Matmul_quant' in node_names)
        # self.assertTrue('add' in node_names)
        # self.assertTrue('add2' in node_names)

    def test_new_API(self):
        import time

        result = [0.1]

        def sub_eval(model, result):
            time.sleep(0.001 * len(result))
            return result[0]

        def eval(model):
            return sub_eval(model, result)

        dataset = Datasets("onnxrt_qdq")["dummy"]([(1, 1, 5, 5), (1, 1, 5, 1)])
        dataloader = DATALOADERS["onnxrt_qdq"](dataset)
        config = PostTrainingQuantConfig(approach="static")
        q_model = quantization.fit(self.matmul_model2, config, calib_dataloader=dataloader, eval_func=eval)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "QLinearMatMul"]), 2)

        config = PostTrainingQuantConfig(approach="static", quant_format="QDQ")
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("QLinearMatMul" not in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="static")
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("QLinearMatMul" in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="dynamic")
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("MatMulInteger" in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="dynamic", quant_format="QDQ")
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("MatMulInteger" in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="static", backend="onnxrt_trt_ep", device="gpu")
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("QLinearMatMul" not in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="static", backend="onnxrt_cuda_ep", device="gpu", quant_level=1)
        q_model = quantization.fit(
            self.distilbert_model,
            config,
            calib_dataloader=DummyNLPDataloader_dict("distilbert-base-uncased-finetuned-sst-2-english"),
            eval_func=eval,
        )
        self.assertTrue("QLinearMatMul" in [i.op_type for i in q_model.nodes()])

        config = PostTrainingQuantConfig(approach="static", recipes={"optypes_to_exclude_output_quant": ["MatMul"]})
        q_model = quantization.fit(self.matmul_model, config, calib_dataloader=self.matmul_dataloader, eval_func=eval)
        self.assertTrue("MatMulIntegerToFloat" in [i.op_type for i in q_model.nodes()])

        dataset = Datasets("onnxrt_qdq")["dummy"]((1, 1), low=0.0, high=0.0, dtype="int64")
        dataloader = DATALOADERS["onnxrt_qdq"](dataset)
        config = PostTrainingQuantConfig()
        q_model = quantization.fit(self.gather_matmul_model, config, calib_dataloader=dataloader, eval_func=eval)

        config = PostTrainingQuantConfig(quant_format="QDQ")
        q_model2 = quantization.fit(self.gather_matmul_model, config, calib_dataloader=dataloader, eval_func=eval)

        sess1 = ort.InferenceSession(q_model.model.SerializeToString(), providers=["CPUExecutionProvider"])
        sess2 = ort.InferenceSession(q_model2.model.SerializeToString(), providers=["CPUExecutionProvider"])
        for data, _ in dataloader:
            output1 = sess1.run(None, {"input0": data})
            output2 = sess2.run(None, {"input0": data})
        self.assertAlmostEqual(output1[0][0], output2[0][0])

    def test_smooth_quant(self):
        config = PostTrainingQuantConfig(
            approach="static", recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        )
        q_model = quantization.fit(self.conv_model, config, calib_dataloader=self.cv_dataloader)
        self.assertEqual(len([i for i in q_model.nodes() if i.op_type == "Mul"]), 2)

    def test_smooth_quant_args(self):
        from neural_compressor.model.onnx_model import ONNXModel

        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "imagenet"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        adaptor.pre_optimized_model = ONNXModel(self.conv_model)
        # tune_cfg was removed, not need to set it to None
        # adaptor.smooth_quant(self.conv_model, self.cv_dataloader, 1, None, scales_per_op=False)
        adaptor.smooth_quant(self.conv_model, self.cv_dataloader, 1, scales_per_op=False)
        self.assertEqual(len([i for i in adaptor.pre_optimized_model.nodes() if i.op_type == "Mul"]), 1)

    def test_multi_metrics(self):
        conf.model.framework = "onnxrt_qlinearops"
        conf.quantization.approach = "post_training_static_quant"
        conf.evaluation.accuracy.multi_metrics = {"Accuracy": {}, "MSE": {"compare_label": False}}
        conf.evaluation.accuracy.pop("metric", None)
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        conf.evaluation.accuracy.multi_metrics = {
            "Accuracy": {},
            "MSE": {"compare_label": False},
            "higher_is_better": [False, False],
        }
        conf.tuning.exit_policy.max_trials = 1
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertEqual(q_model, None)

        conf.tuning.accuracy_criterion.relative = 0.01
        conf.tuning.accuracy_criterion.higher_is_better = True
        conf.evaluation.accuracy.multi_metrics = {"Accuracy": {}, "MSE": {"compare_label": False}, "weight": [0.5, 0.5]}
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        conf.evaluation.accuracy.multi_metrics = {
            "Accuracy": {},
            "MSE": {"compare_label": False},
            "weight": [0.5, 0.5],
            "higher_is_better": [False, False],
        }
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        conf.evaluation.accuracy.multi_metrics = {
            "Accuracy": {},
            "MSE": {"compare_label": False},
            "weight": [0.5, 0.5],
            "higher_is_better": [False, False],
        }
        conf.tuning.accuracy_criterion.higher_is_better = False
        conf.tuning.exit_policy.max_trials = 2
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertEqual(q_model, None)

        import time

        result = [[0.0, 0.0], [0.0, 0.0], [0.0, 122.0]]

        def sub_eval(model, result):
            time.sleep(0.001 * len(result))
            del result[0]
            return result[0]

        def eval(model):
            return sub_eval(model, result)

        conf.evaluation.accuracy.multi_metrics = {
            "Accuracy": {},
            "MSE": {"compare_label": False},
            "higher_is_better": [False, False],
        }
        conf.tuning.exit_policy.max_trials = 1
        conf.tuning.accuracy_criterion = {"absolute": 0.01, "higher_is_better": False}
        from neural_compressor.experimental import Quantization

        quantizer = Quantization(conf)
        quantizer.eval_func = eval
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertEqual(q_model, None)

    def test_calibrator(self):
        from neural_compressor.adaptor.ox_utils.calibrator import CALIBRATOR

        regular_data = [np.arange(15).reshape(3, 5).astype("float32"), np.arange(15).reshape(3, 5).astype("float32")]
        irregular_data = [np.arange(10).reshape(2, 5).astype("float32"), np.arange(5).reshape(1, 5).astype("float32")]

        calibrator = CALIBRATOR["minmax"]()
        calibrator.collect(irregular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.0).astype(np.float32))
        calibrator.collect(regular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(14.0).astype(np.float32))
        calibrator.clear()
        res = calibrator.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calibrator

        calibrator = CALIBRATOR["kl"]()
        calibrator.collect(irregular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.0).astype(np.float32))
        calibrator.collect(regular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(9.140625).astype(np.float32))
        calibrator.clear()
        res = calibrator.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calibrator

        calibrator = CALIBRATOR["percentile"]()
        calibrator.collect(irregular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(8.991211).astype(np.float32))
        calibrator.collect(regular_data)
        res = calibrator.calib_range
        self.assertEqual(res[0], np.array(0.0).astype(np.float32))
        self.assertEqual(res[1], np.array(13.9921875).astype(np.float32))
        calibrator.clear()
        res = calibrator.calib_range
        self.assertIsNone(res[0])
        self.assertIsNone(res[1])
        del calibrator

    def test_query_block_info(self):
        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "nlp"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        q_capability = adaptor.query_fw_capability(Model(self.distilbert_model))
        self.assertEqual(len(q_capability["block_wise"]), 6)

        framework_specific_info = {
            "device": "cpu",
            "approach": "post_training_static_quant",
            "random_seed": 1234,
            "q_dataloader": None,
            "backend": "default",
            "format": "default",
            "domain": "auto",
            "recipes": {},
            "workspace_path": "./nc_workspace/{}/{}/".format("onnxrt", "nlp"),
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        q_capability = adaptor.query_fw_capability(Model(self.albert_model))
        self.assertEqual(len(q_capability["block_wise"]), 12)

    def test_dataloader_input(self):
        cv_dataloader = DataLoader(framework="onnxruntime", dataset=DummyCVDataset_list(shape=(3, 224, 224)))

        quantizer = Quantization("qlinear.yaml")
        quantizer.calib_dataloader = cv_dataloader
        quantizer.eval_dataloader = cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        cv_dataloader = DataLoader(framework="pytorch", dataset=DummyCVDataset_dict(shape=(3, 224, 224)))
        quantizer = Quantization("qlinear.yaml")
        quantizer.calib_dataloader = cv_dataloader
        quantizer.eval_dataloader = cv_dataloader
        quantizer.model = self.rn50_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        nlp_dataloader = DummyNLPDataloader_list("distilbert-base-uncased-finetuned-sst-2-english")
        quantizer = Quantization("qlinear.yaml")
        quantizer.calib_dataloader = nlp_dataloader
        quantizer.eval_dataloader = nlp_dataloader
        quantizer.model = self.distilbert_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

        nlp_dataloader = DummyNLPDataloader_dict("distilbert-base-uncased-finetuned-sst-2-english")
        quantizer = Quantization("qlinear.yaml")
        quantizer.calib_dataloader = nlp_dataloader
        quantizer.eval_dataloader = nlp_dataloader
        quantizer.model = self.distilbert_model
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    @patch("logging.Logger.warning")
    def test_backend(self, mock_warning):
        framework_specific_info = {
            "device": "cpu",
            "backend": "test_backend",
            "approach": "post_training_static_quant",
            "workspace_path": "./nc_workspace",
        }
        framework = "onnxrt_qlinearops"
        with self.assertRaises(AssertionError) as context:
            adaptor = FRAMEWORKS[framework](framework_specific_info)
        self.assertEqual(
            str(context.exception),
            "'test_backend' backend is not supported, "
            "supported backends include ['default', 'onnxrt_trt_ep', 'onnxrt_dnnl_ep', 'onnxrt_cuda_ep', 'onnxrt_dml_ep']",
        )

        framework_specific_info = {
            "device": "cpu",
            "backend": "onnxrt_trt_ep",
            "approach": "post_training_static_quant",
            "workspace_path": "./nc_workspace",
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)

        call_args_list = mock_warning.call_args_list
        first_warning_args = call_args_list[0][0]
        self.assertEqual(first_warning_args[0], "Backend `onnxrt_trt_ep` requires a GPU device. Reset device to 'gpu'.")
        second_warning_args = call_args_list[1][0]
        self.assertIn("not in available provider names. Fallback to available providers", second_warning_args[0])

        self.assertEqual(mock_warning.call_count, 2)

        framework_specific_info = {
            "device": "cpu",
            "backend": "onnxrt_dml_ep",
            "approach": "post_training_static_quant",
            "workspace_path": "./nc_workspace",
        }
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)

        call_args_list = mock_warning.call_args_list
        first_warning_args = call_args_list[2][0]
        self.assertEqual(first_warning_args[0], "Backend `onnxrt_dml_ep` requires a NPU device. Reset device to 'npu'.")
        second_warning_args = call_args_list[3][0]
        self.assertIn("not in available provider names. Fallback to available providers", second_warning_args[0])

    def test_cuda_ep_env_set(self):
        config = PostTrainingQuantConfig(approach="static", backend="onnxrt_cuda_ep", device="gpu", quant_level=1)
        quantization.fit(
            self.distilbert_model,
            config,
            calib_dataloader=DummyNLPDataloader_dict("distilbert-base-uncased-finetuned-sst-2-english"),
        )

        # check TENSORRT is not loaded if backend is not onnxrt_trt_ep
        self.assertEqual(os.environ.get("ORT_TENSORRT_UNAVAILABLE"), "1")

    def test_model_share_init(self):
        config = PostTrainingQuantConfig(approach="static")
        q_model = quantization.fit(self.shared_init_model, config, calib_dataloader=self.cv_dataloader)
        self.assertNotEqual(q_model, None)
        ort.InferenceSession(q_model.model.SerializeToString(), providers=ort.get_available_providers())

        config = PostTrainingQuantConfig(approach="dynamic")
        q_model = quantization.fit(self.shared_init_model, config, calib_dataloader=self.cv_dataloader)
        self.assertNotEqual(q_model, None)
        ort.InferenceSession(q_model.model.SerializeToString(), providers=ort.get_available_providers())

        config = PostTrainingQuantConfig(
            approach="static", quant_format="QDQ", recipes={"add_qdq_pair_to_weight": True}
        )
        q_model = quantization.fit(self.shared_init_model, config, calib_dataloader=self.cv_dataloader)
        self.assertNotEqual(q_model, None)
        ort.InferenceSession(q_model.model.SerializeToString(), providers=ort.get_available_providers())


if __name__ == "__main__":
    unittest.main()
