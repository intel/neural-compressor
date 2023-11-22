import os
import onnx
import torch
import logging
import argparse
import numpy as np
import transformers
from transformers.data import InputFeatures
from neural_compressor.benchmark import fit
from neural_compressor.data import DataLoader
from neural_compressor.config import BenchmarkConfig

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

class GLUEDataset:
    """Dataset used for GLUE."""
    def __init__(self, model, data_dir, model_name_or_path, max_seq_length=128,\
                do_lower_case=True, task='mrpc', model_type='bert', dynamic_length=False,\
                evaluate=True, transform=None, filter=None):
        self.inputs = [inp.name for inp in onnx.load(model).graph.input]
        task = task.lower()
        model_type = model_type.lower()
        assert task in ['mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'mnli', 'wnli', 'sst-2'], 'Unsupported task type'
        assert model_type in ['distilbert', 'bert', 'mobilebert', 'roberta'], 'Unsupported model type'

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        self.dataset = load_and_cache_examples(data_dir, model_name_or_path, \
            max_seq_length, task, model_type, tokenizer, evaluate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = tuple(t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t for t in self.dataset[index])
        return batch[:len(self.inputs)], batch[-1]

def load_and_cache_examples(data_dir, model_name_or_path, max_seq_length, task, model_type, tokenizer, evaluate):
    from torch.utils.data import TensorDataset

    processor = transformers.glue_processors[task]()
    output_mode = transformers.glue_output_modes[task]
    # Load data features from cache or dataset file
    if not os.path.exists("./dataset_cached"):
        os.makedirs("./dataset_cached")
    cached_features_file = os.path.join("./dataset_cached", 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Load features from cached file {}.".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Create features from dataset file at {}.".format(data_dir))
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_dir) if evaluate else \
            processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                task=task,
                                                label_list=label_list,
                                                max_length=max_seq_length,
                                                output_mode=output_mode,
        )
        logger.info("Save features into cached file {}.".format(cached_features_file))
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def convert_examples_to_features(examples, tokenizer, max_length=128, task=None, label_list=None, 
                                 output_mode="classification", pad_token=0, pad_token_segment_id=0, 
                                 mask_padding_with_zero=True,):
    processor = transformers.glue_processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Use label list {} for task {}.".format(label_list, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_length = len(input_ids)
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, \
            "Error with input_ids length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, \
            "Error with attention_mask length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, \
            "Error with token_type_ids length {} vs {}".format(len(token_type_ids), max_length)
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        feats = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label
        )
        features.append(feats)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input_model", type=str, required=True)
    args = parser.parse_args()

    model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english"
    data_path = "./GLUE_DIR/SST-2"
    task = "sst-2"
    batch_size = 1

    # prepare dataset and dataloader
    dataset = GLUEDataset(args.input_model,
                          data_dir=data_path,
                          model_name_or_path=model_name_or_path,
                          model_type="distilbert",
                          task=task)
    dataloader = DataLoader(framework="onnxruntime", dataset=dataset, batch_size=batch_size)

    # run benchmark
    conf = BenchmarkConfig(iteration=100,
                           cores_per_instance=4,
                           num_of_instance=1)
    fit(args.input_model, conf, b_dataloader=dataloader)