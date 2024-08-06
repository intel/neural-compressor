import sys
import os
import math
from tqdm import tqdm
import shortuuid
import json
import re

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from llava.utils import disable_torch_init
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

class CustomDatasetTextVQA(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)

class TextVQAEvaluator(object):
    def __init__(
            self, 
            model, 
            tokenizer, 
            image_processor, 
            image_folder,
            question_file,
            annotation_file, 
            **kwargs
        ):
        super(TextVQAEvaluator, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_folder = image_folder
        self.question_file = question_file
        self.annotation_file = annotation_file
        # follow parameters can be set as default value.
        self.model_name = kwargs.get("model_name", "llava")
        self.conv_mode = kwargs.get("conv_mode", "vicuna_v1")
        self.num_chunks = kwargs.get("num_chunks", 1)
        self.chunk_idx = kwargs.get("chunk_idx", 0)
        self.temperature = kwargs.get("temperature", 0)
        self.top_p = kwargs.get("top_p", None)
        self.num_beams = kwargs.get("num_beams", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 128)

        if 'plain' in self.model_name and 'finetune' not in self.model_name.lower() and 'mmtag' not in self.conv_mode:
            self.conv_mode = self.conv_mode + '_mmtag'
            print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {self.conv_mode}.')

    def create_dataloader(self):
        questions = [json.loads(q) for q in open(os.path.expanduser(self.question_file), "r")]
        questions = get_chunk(questions, self.num_chunks, self.chunk_idx)
        dataset = CustomDatasetTextVQA(questions, self.image_folder, self.tokenizer, self.image_processor, self.model.config, self.conv_mode)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
        return data_loader, questions

    def run_evaluate(self, result_file = None):
        disable_torch_init()
        dataloader, questions = self.create_dataloader()
        result_file = os.path.expanduser(result_file)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        res_file = open(result_file, "w")
        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(dataloader, questions), total=len(questions)):
            idx = line["question_id"]
            cur_prompt = line["text"]

            input_ids = input_ids.to(device='cuda', non_blocking=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True)

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            res_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": self.model_name,
                                    "metadata": {}}) + "\n")
        res_file.close()

    def prompt_processor(self, prompt):
        if prompt.startswith('OCR tokens: '):
            pattern = r"Question: (.*?) Short answer:"
            match = re.search(pattern, prompt, re.DOTALL)
            question = match.group(1)
        elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
            if prompt.startswith('Reference OCR token:'):
                question = prompt.split('\n')[1]
            else:
                question = prompt.split('\n')[0]
        elif len(prompt.split('\n')) == 2:
            question = prompt.split('\n')[0]
        else:
            assert False

        return question.lower()
    
    def calculate_accuracy(self, result_file = None):
        experiment_name = os.path.splitext(os.path.basename(result_file))[0]
        print(experiment_name)
        annotations = json.load(open(self.annotation_file))['data']
        annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
        results = [json.loads(line) for line in open(result_file)]

        pred_list = []
        for result in results:
            annotation = annotations[(result['question_id'], self.prompt_processor(result['prompt']))]
            pred_list.append({
                "pred_answer": result['text'],
                "gt_answers": annotation['answers'],
            })

        evaluator = TextVQAAccuracyEvaluator()
        print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))



# results



# def eval_single(annotation_file, result_file):
#     experiment_name = os.path.splitext(os.path.basename(result_file))[0]
#     print(experiment_name)
#     annotations = json.load(open(annotation_file))['data']
#     annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
#     results = [json.loads(line) for line in open(result_file)]

#     pred_list = []
#     for result in results:
#         annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
#         pred_list.append({
#             "pred_answer": result['text'],
#             "gt_answers": annotation['answers'],
#         })

#     evaluator = TextVQAAccuracyEvaluator()
#     print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


