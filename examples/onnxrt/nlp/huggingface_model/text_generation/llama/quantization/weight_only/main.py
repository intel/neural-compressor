# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation
import os
import onnx
import json
import random
import torch
import logging
import argparse
import numpy as np
from datasets import load_dataset
import onnxruntime as ort
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import LlamaConfig, LlamaTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.WARN)

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--model_path",
    type=str,
    help="Folder path of pre-trained onnx model"
)
parser.add_argument(
    "--benchmark",
    action="store_true", \
    default=False
)
parser.add_argument(
    "--tune",
    action="store_true", \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    "--output_model",
    type=str,
    default=None,
    help="output model path"
)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
)
parser.add_argument(
    "--tokenizer",
    type=str,
    help="pretrained model name or path of tokenizer files",
    default="meta-llama/Llama-2-7b-hf"
)
parser.add_argument(
    "--workspace",
    type=str,
    help="workspace to save intermediate files",
    default="nc_workspace"
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="WOQ_TUNE",
    choices=["WOQ_TUNE", "RTN", "AWQ", "GPTQ"],
    help="weight only algorithm"
)
parser.add_argument(
    "--pad_max",
    default=196,
    type=int,
)
parser.add_argument(
    "--seqlen",
    default=2048,
    type=int,
)
parser.add_argument(
    "--tasks",
    nargs="+",
    default=["winogrande", "copa", "piqa", "rte", "hellaswag", "openbookqa", \
             "lambada_openai", "lambada_standard", "wikitext"],
    type=str,
    help="tasks list for accuracy validation"
)
parser.add_argument(
    "--dataset",
    nargs="?",
    default="NeelNanda/pile-10k",
    const="NeelNanda/pile-10k"
)
parser.add_argument(
    '--mode',
    type=str,
    help="benchmark mode of performance or accuracy"
)
args = parser.parse_args()

# load model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

def tokenize_function(examples):
    example = tokenizer(examples["text"])
    return example

def replace_architectures(json_path):
    # replace 'LLaMATokenizer' to lowercase 'LlamaTokenizer'
    # to avoid bug 'Tokenizer class LLaMATokenizer does not exist or is not currently imported.'
    # refer to https://github.com/huggingface/transformers/issues/22222#issuecomment-1477171703
    with open(json_path, "r") as file:
        data = json.load(file)
        data["architectures"] = ["LlamaForCausalLM"]
        
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def eval_func(model):
    model_dir = model
    if isinstance(model, str) and model.endswith(".onnx"):
        model_dir = os.path.dirname(model)

    replace_architectures(os.path.join(model_dir, "config.json"))

    results = evaluate(
        model="hf-causal",
        model_args="pretrained=" + model_dir + ",tokenizer="+ args.tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        model_format="onnx",
    )

    eval_acc = 0
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
            eval_acc += results["results"][task_name]["word_perplexity"]
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))
            eval_acc += results["results"][task_name]["acc"]

    if len(args.tasks) != 0:
        eval_acc /= len(args.tasks)

    return eval_acc


class KVDataloader:
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder='train'):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset(args.dataset, split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        session = ort.InferenceSession(model_path)
        inputs_names = [input.name for input in session.get_inputs()]
        self.key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0
        self.session = session if self.use_cache else None

    def collate_batch(self, batch):

        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)

    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                ort_input = {}
                ort_input["input_ids"] = input_ids[:, :-1].detach().cpu().numpy().astype("int64")
                ort_input["attention_mask"] = attention_mask[:, :-1].detach().cpu().numpy().astype("int64")
                input_shape = ort_input["input_ids"].shape
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                ort_input["position_ids"] = position_ids.numpy()
                if self.use_cache:
                    # Create dummy past_key_values for decoder first generation step if none given
                    num_attention_heads = config.num_key_value_heads
                    embed_size_per_head = config.hidden_size // config.num_attention_heads
                    shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                    key_or_value = np.zeros(shape, dtype=np.float32)
                    for key_value_input_name in self.key_value_input_names:
                        ort_input[key_value_input_name] = key_or_value

                    outputs = self.session.run(None, ort_input)

                    # regenerate input
                    ort_input['input_ids'] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype('int64')
                    input_shape = ort_input["input_ids"].shape
                    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                    ort_input["position_ids"] = position_ids.numpy()
                    for i in range(int((len(outputs) - 1) / 2)):
                        ort_input['past_key_values.{}.key'.format(i)] = outputs[i*2+1]
                        ort_input['past_key_values.{}.value'.format(i)] = outputs[i*2+2]
                    ort_input['attention_mask'] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

                yield ort_input, last_ind.detach().cpu().numpy()
                
        except StopIteration:
            return


class GPTQDataloader:
    def __init__(self, model_path, batch_size=1, seqlen=2048, sub_folder="train"):
        import random
        random.seed(0)
        self.seqlen = seqlen

        self.batch_size=batch_size
        traindata = load_dataset(args.dataset, split=sub_folder)
        traindata = traindata.map(tokenize_function, batched=True)
        traindata.set_format(type="torch", columns=["input_ids", "attention_mask"])

        session = ort.InferenceSession(model_path)
        inputs_names = [input.name for input in session.get_inputs()]
        self.key_value_input_names = [key for key in inputs_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0
        self.session = session if self.use_cache else None


    def __iter__(self):
        try:
            while True:
                while True:
                    i = random.randint(0, len(self.traindata) - 1)
                    trainenc = self.traindata[i]
                    if trainenc["input_ids"].shape[0] > self.seqlen:
                        break
                i = random.randint(0, trainenc["input_ids"].shape[0] - self.seqlen - 1)
                j = i + self.seqlen
                inp = trainenc["input_ids"][i:j].unsqueeze(0)
                mask = torch.ones(inp.shape)

                ort_input = {}
                ort_input["input_ids"] = inp.detach().cpu().numpy().astype("int64")
                ort_input["attention_mask"] = mask.detach().cpu().numpy().astype("int64")
                input_shape = ort_input["input_ids"].shape
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long).unsqueeze(0).view(-1, input_shape[-1])
                ort_input["position_ids"] = position_ids.numpy()
                if self.use_cache:
                    # create dummy past_key_values for decoder first generation step
                    num_attention_heads = config.num_key_value_heads
                    embed_size_per_head = config.hidden_size // config.num_attention_heads
                    shape = (self.batch_size, num_attention_heads, 0, embed_size_per_head)
                    key_or_value = np.zeros(shape, dtype=np.float32)
                    for key_value_input_name in self.key_value_input_names:
                        ort_input[key_value_input_name] = key_or_value

                    outputs = self.session.run(None, ort_input)

                    # regenerate input
                    ort_input['input_ids'] = inp[:, -1].unsqueeze(0).detach().cpu().numpy().astype('int64')
                    for i in range(int((len(outputs) - 1) / 2)):
                        ort_input['past_key_values.{}.key'.format(i)] = outputs[i*2+1]
                        ort_input['past_key_values.{}.value'.format(i)] = outputs[i*2+2]
                    ort_input['attention_mask'] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')

                yield ort_input, 0

        except StopIteration:
            return

if __name__ == "__main__":
    from neural_compressor import set_workspace
    set_workspace(args.workspace)

    if args.benchmark:
        if args.mode == 'performance':            
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            model_name = "model.onnx" # require optimum >= 1.14.0
            model_path = os.path.join(args.model_path, model_name)
            dataloader = KVDataloader(model_path, pad_max=args.pad_max, batch_size=1)
            conf = BenchmarkConfig(iteration=100,
                                   cores_per_instance=28,
                                   num_of_instance=1)
            fit(model_path, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval_func(args.model_path)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

        

    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig

        model_name = "model.onnx" # require optimum >= 1.14.0
        model_path = os.path.join(args.model_path, model_name)
        if args.algorithm.upper() == "RTN":
            dataloader = KVDataloader(model_path, pad_max=args.pad_max, batch_size=1)
            config = PostTrainingQuantConfig(
                approach="weight_only",
                calibration_sampling_size=[8],
                op_type_dict={".*": {"weight": {"algorithm": ["RTN"]}}},
                )
            q_model = quantization.fit(
                model_path,
                config,
                calib_dataloader=dataloader)

        elif args.algorithm.upper() == "AWQ":
            dataloader = KVDataloader(model_path, pad_max=args.pad_max, batch_size=1)
            config = PostTrainingQuantConfig(
                approach="weight_only",
                calibration_sampling_size=[8],
                recipes={"awq_args": {"enable_mse_search": False}},
                op_type_dict={".*": {"weight": {"algorithm": ["AWQ"]}}},
                )
            q_model = quantization.fit(
                model_path,
                config,
                calib_dataloader=dataloader)

        elif args.algorithm.upper() == "GPTQ":
            dataloader = GPTQDataloader(model_path, seqlen=args.seqlen, batch_size=1)
            config = PostTrainingQuantConfig(
                approach="weight_only",
                calibration_sampling_size=[8],
                op_type_dict={".*": {"weight": {"algorithm": ["GPTQ"], "scheme": ["asym"]}}},
                )
            q_model = quantization.fit(
                model_path,
                config,
                calib_dataloader=dataloader)

        elif args.algorithm.upper() == "WOQ_TUNE":
            from neural_compressor.config import AccuracyCriterion
            dataloader = GPTQDataloader(model_path, seqlen=args.seqlen, batch_size=1)
            accuracy_criterion = AccuracyCriterion(tolerable_loss=0.005)
            config = PostTrainingQuantConfig(
                approach="weight_only",
                calibration_sampling_size=[8],
                accuracy_criterion=accuracy_criterion)
            q_model = quantization.fit(
                model_path,
                config,
                calib_dataloader=dataloader,
                eval_func=eval_func)

        q_model.save(os.path.join(args.output_model, model_name))
