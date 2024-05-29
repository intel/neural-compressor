import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
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

# base evaluator
# class BaseEvaluator(object):
#     def __init__(self, questions = None, images = None):
#         # data related
#         self.question_file = questions # the question file to be loaded
#         self.image_folder = images # images to be loaded
#         self.questions = None
#         self.images = None
#         self.answer_file = None # file to save the output answers
#         # model related
#         self.model = None
#         self.model_name = None
#         self.tokenizer = None
#         self.image_processor = None
#         self.context_len = None
    
#     def prepare_model(self):
#         raise NotImplementedError
    
#     def prepare_data(self):
#         raise NotImplementedError
    
#     def run_inference(self, model):
#         raise NotImplementedError
    
#     def calcualate_benchmark(self, result, annotation):
#         raise NotImplementedError

class TextVQAEvaluator(object):
    def __init__(self, question_file = None, image_folder = None, *args, **kwargs):
        # data related
        self.question_file = question_file # the question file to be loaded
        self.image_folder = image_folder # images to be loaded
        self.questions = None
        self.images = None
        self.answer_file = None # file to save the output answers
        # model related
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.image_processor = None
        self.external_args = kwargs
    
    def prepare_model(self, model_name_or_path = None, model_base = None):
        model_path = os.path.expanduser(model_name_or_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name)

    def prepare_data(self, num_chunks=1, chunk_idx=0, conv_mode = "vicuna_v1"):
        # load textvqa dataloader
        from llava.eval.model_vqa_loader import get_chunk, split_list, collate_fn
        self.questions = [json.loads(q) for q in open(os.path.expanduser(self.question_file), "r")]
        self.questions = get_chunk(self.questions, num_chunks, chunk_idx)
        
        dataset = CustomDataset(self.questions, self.image_folder, self.tokenizer, self.image_processor, self.model.config, conv_mode)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
        return data_loader
    
    def run_inference(self, model_name_or_path, answer_file, temperature = 0):
        self.prepare_model(model_name_or_path)
        data_loader = self.prepare_data()
        self.answer_file = answer_file
        ans_file = open(self.answer_file, "w")
        # run inference
        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, self.questions), total=len(self.questions)):
            idx = line["question_id"]
            cur_prompt = line["text"]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": self.model_name,
                                    "metadata": {}}) + "\n")
        ans_file.close()
    
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

    def calcualate_benchmark(self, answer_file, annotation_file):
        from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
        # load the result files
        experiment_name = os.path.splitext(os.path.basename(answer_file))[0]
        print(experiment_name)
        annotations = json.load(open(annotation_file))['data']
        annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
        results = [json.loads(line) for line in open(answer_file)]

        pred_list = []
        for result in results:
            annotation = annotations[(result['question_id'], self.prompt_processor(result['prompt']))]
            pred_list.append({
                "pred_answer": result['text'],
                "gt_answers": annotation['answers'],
            })

        evaluator = TextVQAAccuracyEvaluator()
        print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))

class POPEEvaluator(object):
    def __init__(self, question_file = None, image_folder = None, *args, **kwargs):
        # data related
        self.question_file = question_file # the question file to be loaded
        self.image_folder = image_folder # images to be loaded
        self.questions = None
        self.images = None
        self.answer_file = None # file to save the output answers
        # model related
        self.model = None
        self.model_name = None
        self.tokenizer = None
        self.image_processor = None
        self.external_args = kwargs
    
    def prepare_model(self, model_name_or_path = None, model_base = None):
        model_path = os.path.expanduser(model_name_or_path)
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, self.model_name)

    def prepare_data(self, num_chunks=1, chunk_idx=0, conv_mode = "vicuna_v1"):
        # load textvqa dataloader
        from llava.eval.model_vqa_loader import get_chunk, split_list, collate_fn
        self.questions = [json.loads(q) for q in open(os.path.expanduser(self.question_file), "r")]
        self.questions = get_chunk(self.questions, num_chunks, chunk_idx)
        
        dataset = CustomDataset(self.questions, self.image_folder, self.tokenizer, self.image_processor, self.model.config, conv_mode)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)
        return data_loader
    
    def run_inference(self, model_name_or_path, answer_file, temperature = 0):
        self.prepare_model(model_name_or_path)
        data_loader = self.prepare_data()
        self.answer_file = answer_file
        ans_file = open(self.answer_file, "w")
        # run inference
        for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, self.questions), total=len(self.questions)):
            idx = line["question_id"]
            cur_prompt = line["text"]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=128,
                    use_cache=True)
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": self.model_name,
                                    "metadata": {}}) + "\n")
        ans_file.close()
    
    def calculate_accuracy(self, answers, label_file):
        label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

        for answer in answers:
            text = answer['text']

            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]

            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answer['text'] = 'no'
            else:
                answer['text'] = 'yes'

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        pred_list = []
        for answer in answers:
            if answer['text'] == 'no':
                pred_list.append(0)
            else:
                pred_list.append(1)

        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2*precision*recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))
        print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    
    def calcualate_benchmark(self, question_file, answer_file, annotation_dir):
        questions = [json.loads(line) for line in open(question_file)]
        questions = {question['question_id']: question for question in questions}
        answers = [json.loads(q) for q in open(answer_file)]
        for file in os.listdir(annotation_dir):
            assert file.startswith('coco_pope_')
            assert file.endswith('.json')
            category = file[10:-5]
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
            self.calculate_accuracy(cur_answers, os.path.join(annotation_dir, file))
            print("====================================")
