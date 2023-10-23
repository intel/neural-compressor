import datasets
import numpy as np
import transformers
import logging
import argparse
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
)

from neural_compressor.config import BenchmarkConfig
from neural_compressor import benchmark

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

task_name = 'mrpc'
raw_datasets = load_dataset("glue", task_name)
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)

sentence1_key, sentence2_key = ("sentence1", "sentence2")
padding = "max_length"
label_to_id = None
max_seq_length = 128
model_name = 'textattack/distilbert-base-uncased-MRPC'

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    finetuning_task=task_name,
    use_auth_token=None,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=None,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    from_tf=False,
    config=config,
    use_auth_token=None,
)

def preprocess_function(examples):
    args = (
        (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    return result

raw_datasets = raw_datasets.map(preprocess_function, batched=True)
eval_dataset = raw_datasets["validation"]
metric = load_metric("glue", task_name)
data_collator = None

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input_model", type=str, required=True)
    args = parser.parse_args()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    eval_dataloader = trainer.get_eval_dataloader()

    b_conf = BenchmarkConfig(warmup=5,
                        iteration=100,
                        cores_per_instance=4,
                        num_of_instance=1)
    benchmark.fit(args.input_model, b_conf, b_dataloader=eval_dataloader)
