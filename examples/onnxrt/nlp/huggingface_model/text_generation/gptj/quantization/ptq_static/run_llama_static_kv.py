import os
import time
import onnx
import torch
import logging
import argparse
import numpy as np
#from transformers import AutoTokenizer
from transformers import LlamaTokenizer
from datasets import load_dataset
import onnxruntime as ort
from torch.nn.functional import pad
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)


parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model_path',
    type=str,
    default='/lfs/opa01/mengniwa/llama-7b-kv/decoder_with_past_model.onnx',
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
    default="/lfs/opa01/mengniwa/nc_workspace"
)
parser.add_argument(
    '--quant_format',
    type=str,
    default='QOperator', 
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
parser.add_argument(
    '--pad_max',
    default=196,
    type=int,
)
args = parser.parse_args()

# load model
#tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(args.model_path))


def tokenize_function(examples):
    example = tokenizer(examples['text'])
    return example

def optimum_eval(model):
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import PretrainedConfig
    dataloader = PTDataloader(pad_max=args.pad_max, batch_size=args.batch_size)
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    sess_options = ort.SessionOptions()
    sessions = ORTModelForCausalLM.load_model(
            os.path.join(model, 'decoder_model.onnx'),
            os.path.join(model, 'decoder_with_past_model.onnx'))
    model = ORTModelForCausalLM(sessions[0], config, model, sessions[1])
    total, hit = 0, 0
    pad_len = 0
    for idx, (batch, last_ind) in enumerate(dataloader):
        label = torch.from_numpy(batch['input_ids'].detach().cpu().numpy().astype('int64')[torch.arange(len(last_ind)), last_ind])
        pad_len = args.pad_max - last_ind - 1

        predictions = model(**batch)
        outputs = predictions[0]

        last_token_logits = outputs[torch.arange(len(last_ind)), -2 - pad_len, :]
        pred = last_token_logits.argmax(dim=-1)
        total += len(label)
        hit += (pred == label).sum().item()
        if idx % 10 == 0:
            print(hit / total)

    acc = hit / total
    return acc

def benchmark(model_path):
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
                             " She wanted to go to places and meet new people, and have fun."
    total_time = 0.0
    num_iter = 10
    num_warmup = 3
    sess_options = ort.SessionOptions()
    #sess_options.intra_op_num_threads = 8
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session1 = ort.InferenceSession(os.path.join(model_path, 'decoder_model.onnx'), sess_options)
    session2 = ort.InferenceSession(os.path.join(model_path, 'decoder_with_past_model.onnx'), sess_options)
    #session1 = ort.InferenceSession('/lfs/opa01/mengniwa/llama-7b-int8/llama_7b_int8.onnx', sess_options)
    #session2 = ort.InferenceSession('/lfs/opa01/mengniwa/llama-7b-kv-int8/decoder_model_with_past_model.onnx', sess_options)
 
    for idx in range(num_iter):
        text = []
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones(input_ids.shape[1])
        attention_mask = attention_mask.unsqueeze(0)
        inp = {'input_ids': input_ids.detach().cpu().numpy(),
                'attention_mask': attention_mask.detach().cpu().numpy().astype('int64')}
        output = session1.run(None, inp)
        logits = output[0]
        logits = torch.from_numpy(logits)
        next_token_logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        for i in range(32):
            inp['past_key_values.{}.key'.format(i)] = output[i*2+1]
            inp['past_key_values.{}.value'.format(i)] = output[i*2+2]
        inp['attention_mask'] =  np.zeros([1, inp['past_key_values.0.key'].shape[2]+1], dtype='int64')
        inp['input_ids'] = np.array([next_tokens.detach().cpu().numpy()], dtype='int64')
        for i in range(32):
            output = session2.run(None, inp)
            logits = output[0]
            logits = torch.from_numpy(logits)
            next_token_logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.argmax(probs, dim=-1)
            text.append(next_tokens)
            print(next_tokens)
            for i in range(32):
                inp['past_key_values.{}.key'.format(i)] = output[i*2+1]
                inp['past_key_values.{}.value'.format(i)] = output[i*2+2]
            inp['attention_mask'] =  np.zeros([1, inp['past_key_values.0.key'].shape[2]+1], dtype='int64')
            inp['input_ids'] = next_tokens.unsqueeze(0).detach().cpu().numpy() 
        toc = time.time()
        print(tokenizer.decode(np.array(text)))
        if idx >= num_warmup:
            total_time += (toc - tic)
    print("Inference latency: %.3f s." % (total_time / (num_iter - num_warmup)))
 
def eval_func(onnx_model, dataloader, workspace, pad_max):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if isinstance(onnx_model, str):
        model_path = onnx_model
    else:
        onnx.save(onnx_model, os.path.join(workspace, 'eval.onnx'), save_as_external_data=True)
        model_path = os.path.join(workspace, 'eval.onnx')

    session = ort.InferenceSession(model_path, options, providers=ort.get_available_providers())
    inputs_names = [i.name for i in session.get_inputs()]

    total, hit = 0, 0
    pad_len = 0

    for idx, (batch, last_ind) in enumerate(dataloader):
        ort_inputs = dict(zip(inputs_names, batch))
        label = torch.from_numpy(batch[0][torch.arange(len(last_ind)), last_ind])
        pad_len = pad_max - last_ind - 1

        predictions = session.run(None, ort_inputs)
        outputs = torch.from_numpy(predictions[0])

        last_token_logits = outputs[torch.arange(len(last_ind)), -2 - pad_len, :]
        pred = last_token_logits.argmax(dim=-1)
        total += len(label)
        hit += (pred == label).sum().item()
        if idx % 10 == 0:
            print(hit / total)

    acc = hit / total
    return acc

def kv_eval_func(onnx_model, dataloader, workspace, pad_max):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if isinstance(onnx_model, str):
        model_path = onnx_model
    else:
        onnx.save(onnx_model, os.path.join(workspace, 'eval.onnx'), save_as_external_data=True)
        model_path = os.path.join(workspace, 'eval.onnx')

    session = ort.InferenceSession(model_path, options, providers=ort.get_available_providers())
    inputs_names = [i.name for i in session.get_inputs()]

    total, hit = 0, 0
    pad_len = 0
    kv = None
    for idx, (batch, last_ind) in enumerate(dataloader):
        ort_inputs = dict(zip(inputs_names, batch))
        if kv == None:
            for i in range(32):
                data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
                data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
 
        label = torch.from_numpy(batch[0][torch.arange(len(last_ind)), last_ind])
        pad_len = pad_max - last_ind - 1

        predictions = session.run(None, ort_inputs)
        outputs = torch.from_numpy(predictions[0])

        last_token_logits = outputs[torch.arange(len(last_ind)), -2 - pad_len, :]
        pred = last_token_logits.argmax(dim=-1)
        total += len(label)
        hit += (pred == label).sum().item()
        if idx % 10 == 0:
            print(hit / total)

    acc = hit / total
    return acc


class PTDataloader:
    def __init__(self, pad_max=196, batch_size=1, sub_folder='validation'):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset('lambada', split=sub_folder)
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
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"] if text["input_ids"].shape[0] <= self.pad_max else text["input_ids"][0:int(self.pad_max-1)]
            #input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            #attention_mask = torch.ones(len(input_ids) + 1)
            #attention_mask[0] = 0
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)

        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)


    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                #data = [input_ids.detach().cpu().numpy().astype('int64')]
                #import pdb;pdb.set_trace()
                #data.append(attention_mask.detach().cpu().numpy().astype('int64'))
                #for i in range(32):
                #    data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
                #    data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
                yield {'input_ids': input_ids.to(torch.int64), 'attention_mask': attention_mask.to(torch.int64)}, last_ind.detach().cpu().numpy()
        except StopIteration:
            return


class Dataloader:
    def __init__(self, pad_max=196, batch_size=1, sub_folder='validation'):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset('lambada', split=sub_folder)
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
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"] if text["input_ids"].shape[0] <= self.pad_max else text["input_ids"][0:int(self.pad_max-1)]
            #input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            #attention_mask = torch.ones(len(input_ids) + 1)
            #attention_mask[0] = 0
            attention_mask = torch.ones(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=1)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)

        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)


    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                data = [input_ids.detach().cpu().numpy().astype('int64')]
                #import pdb;pdb.set_trace()
                data.append(attention_mask.detach().cpu().numpy().astype('int64'))
                #for i in range(32):
                #    data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
                #    data.append(np.zeros((input_ids.shape[0],32,1,128), dtype='float32'))
                yield data, last_ind.detach().cpu().numpy()
        except StopIteration:
            return

class KVDataloader:
    def __init__(self, model_path, pad_max=196, batch_size=1, sub_folder='validation'):
        self.pad_max = pad_max
        self.batch_size=batch_size
        dataset = load_dataset('lambada', split=sub_folder)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        self.sess = None
        if not model_path.endswith('decoder_model.onnx'):
            self.sess = ort.InferenceSession(os.path.join(os.path.dirname(model_path), 'decoder_model.onnx'))


    def collate_batch(self, batch):

        input_ids_padded = []
        attention_mask_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"] if text["input_ids"].shape[0] <= self.pad_max else text["input_ids"][0:int(self.pad_max-1)]
            #input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            #attention_mask = torch.ones(len(input_ids) + 1)
            #attention_mask[0] = 0
            attention_mask = torch.ones(len(input_ids))
            #input_ids = pad(input_ids, (0, pad_len), value=1)
            input_ids_padded.append(input_ids)
            #attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)

        return (torch.vstack(input_ids_padded), torch.vstack(attention_mask_padded)), torch.tensor(last_ind)


    def __iter__(self):
        try:
            for (input_ids, attention_mask), last_ind in self.dataloader:
                if self.sess is None:
                    yield {'input_ids': input_ids[:, :-1].detach().cpu().numpy().astype('int64'), 'attention_mask':attention_mask[:, :-1].detach().cpu().numpy().astype('int64')}, last_ind.detach().cpu().numpy()
                else:
                    outputs = self.sess.run(None, {'input_ids': input_ids[:, :-1].detach().cpu().numpy().astype('int64'), 'attention_mask':attention_mask[:, :-1].detach().cpu().numpy().astype('int64')})
                    ort_input = {}
                    ort_input['input_ids'] = input_ids[:, -1].unsqueeze(0).detach().cpu().numpy().astype('int64')
                    for i in range(32):
                        ort_input['past_key_values.{}.key'.format(i)] = outputs[i*2+1]
                        ort_input['past_key_values.{}.value'.format(i)] = outputs[i*2+2]
                    ort_input['attention_mask'] =  np.zeros([self.batch_size, ort_input['past_key_values.0.key'].shape[2]+1], dtype='int64')
                    yield ort_input, last_ind.detach().cpu().numpy()
        except StopIteration:
            return


if __name__ == "__main__":

    #dataloader = Dataloader(pad_max=args.pad_max, batch_size=args.batch_size)
    def eval(model):
        return eval_func(model, dataloader, args.workspace, args.pad_max)

    if args.benchmark:
        if args.mode == 'performance':            
            #from neural_compressor.benchmark import fit
            #from neural_compressor.config import BenchmarkConfig
            #conf = BenchmarkConfig(iteration=10,
            #                       cores_per_instance=8,
            #                       num_of_instance=1)
            #fit(args.model_path, conf, b_dataloader=dataloader)
            benchmark(args.model_path)
        elif args.mode == 'accuracy':
            #optimum_eval(args.model_path)
            acc_result = eval(args.model_path)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        from neural_compressor import set_workspace
        from neural_compressor import quantization, PostTrainingQuantConfig
        set_workspace(args.workspace)

        config = PostTrainingQuantConfig(
            #approach='dynamic',
            quant_format='QDQ',
            calibration_sampling_size=[8],
            recipes={'optypes_to_exclude_output_quant': ['MatMul'],
                     'smooth_quant': True,
                     'smooth_quant_args': {'alpha': 0.5}
                     },
            op_type_dict={'^((?!(MatMul|Gather|Conv)).)*$': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}},)
            #op_name_dict={'/model/layers.2/mlp/down_proj/MatMul': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
            #              '/model/layers.30/mlp/down_proj/MatMul': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
            #              '/model/layers.0/mlp/down_proj/MatMul': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}},
            #              '/model/layers.1/mlp/down_proj/MatMul': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}})
        q_model = quantization.fit(args.model_path, config, calib_dataloader=KVDataloader(args.model_path, batch_size=1, sub_folder='train'))
        q_model.save(args.output_model)
        #eval(args.output_model)
        #if args.quant_format == 'QOperator':
        #    sess_options = ort.SessionOptions()
        #    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #    sess_options.optimized_model_filepath = args.output_model
        #    ort.InferenceSession(os.path.join(args.workspace, 'eval.onnx'), sess_options, providers=ort.get_available_providers())
        #else:
        #    q_model.save(args.output_model)
