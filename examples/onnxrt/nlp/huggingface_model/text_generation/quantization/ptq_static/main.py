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
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import onnxruntime as ort
from torch.nn.functional import pad

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model_path',
    type=str,
    help="Pre-trained resnet50 model on onnx file"
)
parser.add_argument(
    '--benchmark',
    action='store_true', \
    default=False
)
parser.add_argument(
    '--tune',
    action='store_true', \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    '--output_model',
    type=str,
    default=None,
    help="output model path"
)
parser.add_argument(
    '--mode',
    type=str,
    help="benchmark mode of performance or accuracy"
)
parser.add_argument(
    '--batch_size',
    default=8,
    type=int,
)
parser.add_argument(
    '--model_name_or_path',
    type=str,
    help="pretrained model name or path",
    default="EleutherAI/gpt-j-6B"
)
parser.add_argument(
    '--workspace',
    type=str,
    help="workspace to save intermediate files",
    default="nc_workspace"
)
parser.add_argument(
    '--quant_format',
    type=str,
    default='QOperator', 
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
args = parser.parse_args()

# load model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

def tokenize_function(examples):
    example = tokenizer(examples['text'])
    return example

def eval_func(onnx_model, dataloader, workspace):
    options = ort.SessionOptions()
    onnx.save(onnx_model, os.path.join(workspace, 'eval.onnx'), save_as_external_data=True)
    session = ort.InferenceSession(os.path.join(workspace, 'eval.onnx'), options,
        providers=ort.get_available_providers())
    inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]

    total, hit = 0, 0
 
    for idx, batch in enumerate(dataloader):
        ort_inputs = dict(zip(inputs_names, batch))
        predictions = session.run(None, ort_inputs)
        outputs = torch.from_numpy(predictions[0]) 
        last_token_logits = outputs[:, -2 - pad_len, :]
        pred = last_token_logits.argmax(dim=-1)
        total += label.size(0)
        hit += (pred == label).sum().item()
    
    acc = hit / total
    return acc

class Dataloader:
    def __init__(self, pad_max=196, batch_size=8):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset('lambada', split='validation')
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch):

        input_ids_padded = []
        attention_mask_padded = []

        for text in batch:
            input_ids = text["input_ids"]
            attention_mask = text["attention_mask"]
            pad_len = 196 - len(input_ids)

            input_ids = pad(input_ids, (0, pad_len), value=1)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=1)
            attention_mask_padded.append(attention_mask)

        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded))


    def __iter__(self):
        try:
            for input_ids, attention_mask in self.dataloader:
                data = [input_ids.detach().cpu().numpy().astype('int64')]
                for i in range(28):
                    data.append(np.zeros((input_ids.shape[0],16,1,256), dtype='float32'))
                    data.append(np.zeros((input_ids.shape[0],16,1,256), dtype='float32'))
                attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1] +1))
                attention_mask[:,0] = 0
                data.append(attention_mask.detach().cpu().numpy().astype('int64'))
                yield data, 1
        except StopIteration:
            return
            
if __name__ == "__main__":
    from neural_compressor import set_workspace
    set_workspace(args.workspace)

    dataloader = Dataloader(args.batch_size)
    def eval(model):
        return eval_func(model, dataloader, args.workspace)

    if args.benchmark:
        model = onnx.load(args.model_path)
        if args.mode == 'performance':            
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(iteration=100,
                                   cores_per_instance=28,
                                   num_of_instance=1)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        import onnx
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(
            quant_format='QDQ',
            calibration_sampling_size=[8],
            recipes={'optypes_to_exclude_output_quant': ['MatMul']},
            op_type_dict={'^((?!(MatMul|Gather)).)*$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}})
        q_model = quantization.fit(args.model_path, config, eval_func=eval, calib_dataloader=dataloader)
        q_model.save(args.output_model)
