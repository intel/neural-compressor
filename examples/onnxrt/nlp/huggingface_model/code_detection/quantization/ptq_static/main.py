from __future__ import absolute_import, division, print_function

import argparse
import logging
from typing import List

import numpy as np
import onnx

logger = logging.getLogger(__name__)


def load_dataset_from_local(file_path, model_name_or_path):
    """Load the raw data from local."""
    import json

    import torch

    def read_data(file_path):
        texts, labels = [], []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                js = json.loads(line.strip())
                code = " ".join(js["func"].split())
                texts.append(code)
                labels.append(js["target"])
        return texts, labels

    texts, labels = read_data(file_path)

    # tokenize the raw data
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True, padding="max_length"
    )

    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = CodeDataset(encodings, labels)
    return dataset


# evaluation func for fine-tuning
def evaluate(model, val_loader):
    import torch

    print("***　eval model .. ")
    all_labels = []
    all_preds = []
    for idx, batch in enumerate(val_loader):
        model.eval()
        with torch.no_grad():
            labels = batch.pop("labels")
            inputs = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            all_labels.append(labels.numpy())
            all_preds.append(np.argmax(logits.detach().numpy(), axis=1))
            np.concatenate(all_labels, axis=0)
            np.concatenate(all_preds, axis=0)
            cur_acc = np.mean(
                np.concatenate(all_labels, axis=0) == np.concatenate(all_preds, axis=0)
            )
            print(f"{idx} batch evaluation accuracy: {cur_acc}")
    cur_acc = np.mean(
        np.concatenate(all_labels, axis=0) == np.concatenate(all_preds, axis=0)
    )
    print("Overall evaluation accuracy: ", cur_acc)
    return cur_acc


def fine_tune(args):
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AdamW, AutoModelForSequenceClassification

    train_dataset = load_dataset_from_local(
        args.train_data_path, args.model_name_or_path
    )
    val_dataset = load_dataset_from_local(args.data_path, args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    results = {"eval_acc": 0}
    global_step = -1
    for epoch in range(3):
        all_labels = []
        all_preds = []
        for idx, batch in enumerate(train_loader):
            global_step += 1
            optim.zero_grad()
            labels = batch.pop("labels")
            inputs = batch
            model.train()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            all_labels.append(labels.numpy())
            all_preds.append(np.argmax(logits.detach().numpy(), axis=1))
            np.concatenate(all_labels, axis=0)
            np.concatenate(all_preds, axis=0)
            cur_acc = np.mean(
                np.concatenate(all_labels, axis=0) == np.concatenate(all_preds, axis=0)
            )
            print("  Current acc:%s", round(cur_acc, 4))
            loss.backward()
            print(f" Loss: {loss.item()}")
            optim.step()

            if global_step % 100 == 0:
                best_acc = results["eval_acc"]
                cur_acc = evaluate(model, val_loader)
                if cur_acc > best_acc:
                    results["eval_acc"] = cur_acc
                    best_acc = results["eval_acc"]
                    print("  Best acc:%s", round(best_acc, 4))
                    checkpoint_prefix = "checkpoint-best-acc"
                    output_dir = os.path.join("{}".format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model.config.to_json_file("{}/config.json".format(checkpoint_prefix))
                    output_dir = os.path.join(output_dir, "{}".format("pytorch_model.bin"))
                    torch.save(model_to_save.state_dict(), output_dir)
                    print("Saving model checkpoint to %s", output_dir)


class ONNXRTDataset:
    def __init__(self, model_path, dataset):
        self.inputs = [inp.name for inp in onnx.load(model_path).graph.input]
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = self.dataset[index]
        labels = batch["labels"].detach().cpu().numpy()
        batch.pop("labels")
        inputs = [batch["input_ids"].numpy(), batch["attention_mask"].numpy()]
        return inputs, labels


def get_dataloader(ort_model_path, dataset):
    """Create INC ORT dataloader."""
    dataloader = ONNXRTDataset(ort_model_path, dataset)
    return dataloader


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        help="An optional input training data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--model_path", default=None, type=str, help="The onnx model path."
    )
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        default=False,
        help="whether fine tune the model",
    )
    parser.add_argument(
        "--tune", action="store_true", default=False, help="whether quantize the model"
    )
    parser.add_argument(
        "--output_model", type=str, default=None, help="output model path"
    )
    parser.add_argument(
        "--mode", type=str, help="benchmark mode of performance or accuracy"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--quant_format",
        type=str,
        default="QOperator",
        choices=["QOperator", "QDQ"],
        help="quantization format",
    )
    args = parser.parse_args()

    # fine tune
    if args.fine_tune:
        fine_tune(args)


    def eval_func(model):
        session = ort.InferenceSession(
            model.SerializeToString(), providers=ort.get_available_providers()
        )
        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        all_labels = []
        all_preds = []
        import tqdm

        for idx, (inputs, labels) in tqdm.tqdm(enumerate(dataloader)):
            if not isinstance(labels, list):
                labels: List[np.array] = [labels]  # List[shape: bs]
            inputs = inputs[:len_inputs]
            for i in range(len_inputs):
                ort_inputs.update({inputs_names[i]: inputs[i]})
            predictions: List[np.array] = session.run(
                None, ort_inputs
            )  # List[# shape, (bs, 2)]
            predictions = [np.argmax(p, axis=1) for p in predictions]

            all_labels += labels
            all_preds += predictions
            np.mean(
                np.concatenate(all_labels, 0) == (np.concatenate(all_preds, 0))
            )  # [:,0]>0.5))
        label_flatten = np.concatenate(all_labels, 0)
        preds_flatten = np.concatenate(all_preds, 0)
        correct_count = np.sum(label_flatten == preds_flatten)
        acc = correct_count / len(label_flatten)
        return acc
    
    # tune
    if args.tune:
        from neural_compressor import PostTrainingQuantConfig, quantization
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions

        train_dataset = load_dataset_from_local(args.data_path, args.model_name_or_path)
        ort_dataset = ONNXRTDataset(args.model_path, train_dataset)

        from neural_compressor.data import DataLoader as INC_DataLoader

        dataloader = INC_DataLoader(
            framework="onnxruntime", dataset=ort_dataset, batch_size=args.batch_size
        )

        model_type = "bert"
        opt_options = FusionOptions(model_type)
        opt_options.enable_embed_layer_norm = False

        model_optimizer = optimizer.optimize_model(
            args.model_path,
            model_type,
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options,
        )
        model = model_optimizer.model

        # check the optimized model is valid
        import onnxruntime as ort

        try:
            ort.InferenceSession(
                model.SerializeToString(), providers=ort.get_available_providers()
            )
        except Exception as e:
            logger.warning("Optimized model is invalid: {}. ".format(e))
            logger.warning(
                "Model optimizer will be skipped. "
                "Try to upgrade onnxruntime to avoid this error"
            )
            model = onnx.load(args.model_path)

        config = PostTrainingQuantConfig(
            approach="static",
            quant_level=1,
            quant_format=args.quant_format,
        )
        q_model = quantization.fit(
            model,
            config,
            eval_func=eval_func,
            calib_dataloader=dataloader,
        )
        q_model.save(args.output_model)

    # benchmark
    if args.benchmark:
        import onnx
        import onnxruntime as ort
        from neural_compressor.data import DataLoader as INC_DataLoader

        train_dataset = load_dataset_from_local(args.data_path, args.model_name_or_path)
        ort_dataset = ONNXRTDataset(args.model_path, train_dataset)
        dataloader = INC_DataLoader(
            framework="onnxruntime", dataset=ort_dataset, batch_size=args.batch_size
        )
        model = onnx.load(args.model_path)
        if args.mode == "performance":
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig

            conf = BenchmarkConfig(iteration=100)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == "accuracy":
            acc_result = eval_func(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)


if __name__ == "__main__":
    main()
