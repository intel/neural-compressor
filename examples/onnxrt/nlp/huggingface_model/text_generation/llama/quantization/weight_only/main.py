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
    default="RTN",
    choices=["RTN", "AWQ", "GPTQ"],
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
args = parser.parse_args()

# load model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

def tokenize_function(examples):
    example = tokenizer(examples["text"])
    return example

def eval_func(model):
    results = evaluate(
        model="hf-causal",
        model_args="pretrained=" + model + ",tokenizer="+ args.tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        model_format="onnx"
    )
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]))
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]))

class KVDataloader:
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder="train"):
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
        self.sess = None
        if not model_path.endswith("decoder_model.onnx"):
            self.sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), "decoder_model.onnx"))


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
                if self.sess is None:
                    yield {"input_ids": input_ids[:, :-1].detach().cpu().numpy().astype("int64"),
                           "attention_mask":attention_mask[:, :-1].detach().cpu().numpy().astype("int64")}, last_ind.detach().cpu().numpy()
                else:
                    outputs = self.sess.run(None, {"input_ids": input_ids[:, :-1].detach().cpu().numpy().astype("int64"),
                                                   "attention_mask":attention_mask[:, :-1].detach().cpu().numpy().astype("int64")})
                    ort_input = {}
                    ort_input["input_ids"] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype("int64")
                    for i in range(int((len(outputs) - 1) / 2)):
                        ort_input["past_key_values.{}.key".format(i)] = outputs[i*2+1]
                        ort_input["past_key_values.{}.value".format(i)] = outputs[i*2+2]
                    ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input["past_key_values.0.key"].shape[2]+1], dtype="int64")
                    yield ort_input, last_ind.detach().cpu().numpy()
        except StopIteration:
            return

class GPTQDataloader:
    def __init__(self, model_path, batch_size=1, seqlen=2048, sub_folder="train"):
        import random
        random.seed(0)
        self.seqlen = seqlen

        self.batch_size=batch_size
        self.traindata = load_dataset(args.dataset, split=sub_folder)
        self.traindata = self.traindata.map(tokenize_function, batched=True)
        self.traindata.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.sess = None
        if not model_path.endswith("decoder_model.onnx"):
            self.sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), "decoder_model.onnx"))

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
                if self.sess is None:
                    yield {"input_ids": inp.detach().cpu().numpy().astype("int64"),
                        "attention_mask": mask.detach().cpu().numpy().astype("int64")}, 0
                else:
                    outputs = self.sess.run(None, {"input_ids": inp[:, :-1].detach().cpu().numpy().astype("int64"),
                                                   "attention_mask": mask[:, :-1].detach().cpu().numpy().astype("int64")})
                    ort_input = {}
                    ort_input["input_ids"] = inp[:, -1].unsqueeze(0).detach().cpu().numpy().astype("int64")
                    for i in range(int((len(outputs) - 1) / 2)):
                        ort_input["past_key_values.{}.key".format(i)] = outputs[i*2+1]
                        ort_input["past_key_values.{}.value".format(i)] = outputs[i*2+2]
                    ort_input["attention_mask"] =  np.zeros([self.batch_size, ort_input["past_key_values.0.key"].shape[2]+1], dtype="int64")
                    yield ort_input, 0
 
        except StopIteration:
            return

if __name__ == "__main__":
    from neural_compressor import set_workspace
    set_workspace(args.workspace)

    if args.benchmark:
        eval_func(args.model_path)

    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        for model in ["decoder_model.onnx", "decoder_with_past_model.onnx"]:
            if args.algorithm.upper() == "RTN":
                dataloader = KVDataloader(os.path.join(args.model_path, model), pad_max=args.pad_max, batch_size=1)
                config = PostTrainingQuantConfig(
                    approach="weight_only",
                    calibration_sampling_size=[8],
                    op_type_dict={".*": {"weight": {"algorithm": ["RTN"]}}},
                    )

            elif args.algorithm.upper() == "AWQ":
                dataloader = KVDataloader(os.path.join(args.model_path, model), pad_max=args.pad_max, batch_size=1)
                config = PostTrainingQuantConfig(
                    approach="weight_only",
                    calibration_sampling_size=[8],
                    recipes={"awq_args": {"enable_mse_search": False}},
                    op_type_dict={".*": {"weight": {"algorithm": ["AWQ"]}}},
                    )
 
            elif args.algorithm.upper() == "GPTQ":
                dataloader = GPTQDataloader(os.path.join(args.model_path, model), seqlen=args.seqlen, batch_size=1)
                config = PostTrainingQuantConfig(
                    approach="weight_only",
                    calibration_sampling_size=[8],
                    op_type_dict={".*": {"weight": {"algorithm": ["GPTQ"], "scheme": ["asym"]}}},
                    )

            q_model = quantization.fit(
                    os.path.join(args.model_path, model),
                    config,
                    calib_dataloader=dataloader)
            q_model.save(os.path.join(args.output_model, model))
