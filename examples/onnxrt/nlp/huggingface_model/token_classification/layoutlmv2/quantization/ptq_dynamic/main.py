#!/usr/bin/env python
# coding=utf-8
from image_utils import (
    Compose,
    RandomResizedCropAndInterpolationWithTwoPic,
    pil_loader,
)
from torchvision import transforms
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from neural_compressor.data import DataLoader
import torch
import onnxruntime
import onnx
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import transformers
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed.
# Remove at your own risks.
check_min_version("4.5.0")


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    input_model: str = field(default=None, metadata={"help": "Path to onnx model"})
    tune: bool = field(
        default=False,
        metadata={"help": ("INC tune")},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": ("INC benchmark")},
    )
    mode: str = field(
        default="performance",
        metadata={"help": ("INC benchmark mode")},
    )
    save_path: str = field(
        default=None,
        metadata={"help": ("onnx int8 model path")},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": ("batch size for benchmark")},
    )
    quant_format: str = field(
        default="QOperator",
        metadata={"help": ("quant format")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default="funsd",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default="bicubic",
        metadata={"help": "Training interpolation (random, bilinear, bicubic)"},
    )
    second_interpolation: str = field(
        default="lanczos",
        metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"},
    )
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


class IncDataset:
    def __init__(
        self,
        dataset,
        model,
        label_names=None,
    ):
        self.dataset = dataset
        self.label_names = ["labels"] if label_names is None else label_names
        self.session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=onnxruntime.get_available_providers(),
        )
        self.onnx_input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self._process_dataset()

    def _process_dataset(self):
        self.label = []
        self.onnx_inputs = []
        for inputs in self.dataset:
            onnx_inputs = []
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            self.label.append(labels)
            """
            LayoutLMV2 inputs (with order):
            {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'bbox': {0: 'batch_size', 1: 'sequence_length'},
                'image': {0: 'batch_size', 1: 'num_channels'}, # dtype is np.int64 not float
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            }
            """
            for key in self.onnx_input_names:
                if key in inputs:
                    # onnx_inputs[key] = np.array([inputs[key]])
                    onnx_inputs.append(np.array(inputs[key]))
                    # self.onnx_inputs.append([np.array(inputs[key])])
                elif key == "image":
                    # onnx_inputs[key] = np.array([inputs['images']], dtype=np.float32)
                    onnx_inputs.append(np.array(inputs["images"], dtype=np.int64))
                    # self.onnx_inputs.append([np.array(inputs['images'], dtype=np.float32)])

            self.onnx_inputs.append(onnx_inputs)
            onnx_inputs = []

    def __getitem__(self, index):
        return tuple(self.onnx_inputs[index]), self.label[index]

    def __len__(self):
        assert len(self.label) == len(self.onnx_inputs)
        return len(self.onnx_inputs)


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == "funsd":
        # datasets = load_dataset("nielsr/funsd")
        import funsd

        datasets = load_dataset(os.path.abspath(funsd.__file__), cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    column_names = datasets["test"].column_names
    features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"
    boxes_column_name = "bboxes"

    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = sorted(unique_labels)
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,
        # avoid loading from a cached file of the pre-trained model in another
        # machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose(
            [
                # transforms.ColorJitter(0.4, 0.4, 0.4),
                # transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=data_args.input_size,
                    interpolation=data_args.train_interpolation,
                ),
            ]
        )

        patch_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            boxes=examples[boxes_column_name],
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    validation_name = "test"
    if validation_name not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = datasets[validation_name]
    if data_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_columns,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Evaluation
    from model import ORTModel

    def eval_func(model):
        logger.info("*** Evaluate ***")
        ort_model = ORTModel(
            model,
            compute_metrics=compute_metrics,
        )
        outputs = ort_model.evaluation_loop(eval_dataset)
        return outputs.metrics["f1"]

    if model_args.tune:
        from neural_compressor import PostTrainingQuantConfig, quantization
        from neural_compressor.utils.constant import FP32
        import onnx

        onnx_model = onnx.load(model_args.input_model)
        calib_dataset = IncDataset(eval_dataset, onnx_model)
        # TODO double check it for better perf
        config = PostTrainingQuantConfig(
            approach="dynamic",
            quant_level=1,
            quant_format=model_args.quant_format,
        )
        # recipes={'smooth_quant': True, 'smooth_quant_args': {'alpha': 0.5}},
        # TODO double-check the sq, some issues
        q_model = quantization.fit(
            onnx_model,
            config,
            eval_func=eval_func,
            calib_dataloader=DataLoader(framework="onnxruntime", dataset=calib_dataset, batch_size=1),
        )
        q_model.save(model_args.save_path)

    if model_args.benchmark:
        import onnx
        onnx_model = onnx.load(model_args.input_model)
        if model_args.mode == "performance":
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig

            b_dataset = IncDataset(eval_dataset, onnx_model)
            conf = BenchmarkConfig(
                iteration=100,
                cores_per_instance=28,
                num_of_instance=1,
            )
            b_dataloader = DataLoader(
                framework="onnxruntime",
                dataset=b_dataset,
                batch_size=model_args.batch_size,
            )
            fit(onnx_model, conf, b_dataloader=b_dataloader)
        elif model_args.mode == "accuracy":
            eval_f1 = eval_func(onnx_model)
            print("Batch size = %d" % model_args.batch_size)
            print("Accuracy: %.5f" % eval_f1)


if __name__ == "__main__":
    main()
