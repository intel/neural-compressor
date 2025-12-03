import logging
import os
import sys
from dataclasses import dataclass, field
from warnings import warn

import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
    TrainerCallback,
)

from utils import (
    get_metrics_with_perplexity,
    make_supervised_data_module,
    QATTrainer
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            )
        },
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)

@dataclass
class DataArguments:
    dataset: str = field(
        default="Daring-Anteater",
        metadata={"help": "Specify the dataset.", "choices": ["Daring-Anteater"]},
    )
    train_size: int = field(
        default=0,
        metadata={"help": "Number of training samples to use. If `0`, use default training size."},
    )
    eval_size: int = field(
        default=0,
        metadata={
            "help": "Number of evaluation samples to use. If `0`, use default evaluation size."
        },
    )

@dataclass
class QuantizationArguments:
    quant_scheme: str | None = field(
        default=None,
        metadata={
            "help": (
                "Specify the quantization format for PTQ/QAT. if specified, PTQ/QAT will be enabled"
                " with the specified quantization format"
            ),
            "choices": ["MXFP8", "MXFP4"],
        },
    )


def train():
    parser = HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, QuantizationArguments)
    )

    model_args, training_args, data_args, quant_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info(f"arguments: {model_args}, {training_args}, {data_args}, {quant_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info(f"Last checkpoint detected: {last_checkpoint}")


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.do_sample = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # We set model.config.use_cache to False for training when gradient_checkpointing=False.
    # Currently useful for FSDP2 to allow for setting activation_checkpointing=True in the config file.
    model.config.use_cache = False

    # prepare model for quantization
    if quant_args.quant_scheme is not None:
        from neural_compressor.torch.quantization.quantize import prepare_qat

        model.train()
        # inplace
        if quant_args.quant_scheme == "MXFP8":
            # default mxfp8
            prepare_qat(model)
        if quant_args.quant_scheme == "MXFP4":
            mappings = {torch.nn.Linear: "MXFP4"}
            prepare_qat(model, mappings)


        logger.info("Finish model preparation for QAT.")

    logger.info("Loading dataset......")

    # reuse the dataset function, TODO: preprocess a new dataset
    data_module = make_supervised_data_module(
        dataset=data_args.dataset,
        tokenizer=tokenizer,
        train_size=data_args.train_size,
        eval_size=data_args.eval_size,
    )

    # Ensure calibration size doesn't exceed evaluation dataset size
    eval_dataset_size = len(data_module["eval_dataset"])

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Torch >= 2.4 throws an error if `use_reentrant` is not set explicitly
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    trainer = QATTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if training_args.do_train:
        logger.info("Starting Train...")
        trainer.train(resume_from_checkpoint=checkpoint)
        logger.info("Training completed.")

    if training_args.do_eval:
        logger.info("Starting Evaluation...")
        metrics = trainer.evaluate()
        metrics = get_metrics_with_perplexity(metrics)
        logger.info(f"Evaluation results: \n{metrics}")

    logger.info("Saving the model...")
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
