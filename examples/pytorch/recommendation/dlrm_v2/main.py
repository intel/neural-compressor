import argparse
import itertools
import numpy as np
import sys
from torch.profiler import record_function
from pprint import pprint
from typing import List
import time

import torch

import torchmetrics as metrics
from pyre_extensions import none_throws
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.models.dlrm import DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from tqdm import tqdm
from neural_compressor.torch.quantization import (
    FP8Config,
    convert,
    finalize_calibration,
    prepare,
)
from dlrm_model import OPTIMIZED_DLRM_DCN, replace_crossnet
from data_process.dlrm_dataloader import get_dataloader


TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


def unpack(input: KeyedJaggedTensor) -> dict:
    output = {}
    for k, v in input.to_dict().items():
        output[k] = {}
        output[k]["values"] = v._values.int()
        output[k]["offsets"] = v._offsets.int()
    return output


def load_snapshot(model, model_path):
    from torchsnapshot import Snapshot

    snapshot = Snapshot(path=model_path)
    snapshot.restore(app_state={"model": model})


def fetch_batch(dataloader):
    try:
        batch = dataloader.dataset.load_batch()
    except:
        import torchrec

        dataset = dataloader.source.dataset
        if isinstance(
            dataset, torchrec.datasets.criteo.InMemoryBinaryCriteoIterDataPipe
        ):
            sample_list = list(range(dataset.batch_size))
            dense = dataset.dense_arrs[0][sample_list, :]
            sparse = [arr[sample_list, :] for arr in dataset.sparse_arrs][
                0
            ] % dataset.hashes
            labels = dataset.labels_arrs[0][sample_list, :]
            return dataloader.func(dataset._np_arrays_to_batch(dense, sparse, labels))
        batch = dataloader.func(
            dataloader.source.dataset.batch_generator._generate_batch()
        )
    return batch


class DLRM_DataLoader(object):
    def __init__(self, loader=None):
        self.loader = loader
        self.batch_size = 1
    def __iter__(self):
        for dense_feature, sparse_dfeature in [(self.loader.dense_features, self.loader.sparse_features)]:
            yield {"dense_features": dense_feature, "sparse_features": sparse_dfeature}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=100,
        help="number of validation batches",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=100,
        help="number of test batches",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1696543516,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--inductor",
        action="store_true",
        help="whether use torch.compile()",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )

    parser.add_argument("--calib", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--out_dir", type=str, default="inc_fp8/measure", help="A folder to save calibration result")
    return parser.parse_args(argv)


def _evaluate(
    eval_model,
    eval_dataloader: DataLoader,
    stage: str,
    args,
) -> float:
    """
    Evaluates model. Computes and prints AUROC

    Args:
        model (torch.nn.Module): model for evaluation.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".
        args (argparse.Namespace): parsed command line args.
        
    Returns:
        float: auroc result
    """
    limit_batches = args.limit_val_batches

    benckmark_batch = fetch_batch(eval_dataloader)
    benckmark_batch.sparse_features = unpack(benckmark_batch.sparse_features)
    def fetch_next(iterator, current_it):
        with record_function("generate batch"):
            next_batch = next(iterator)
        with record_function("unpack KeyJaggedTensor"):
            next_batch.sparse_features = unpack(next_batch.sparse_features)
        return next_batch

    def eval_step(model, iterator, current_it):
        batch = fetch_next(iterator, current_it)
        with record_function("model forward"):
            t1 = time.time()
            logits = model(batch.dense_features, batch.sparse_features)
            t2 = time.time()
        return logits, batch.labels, t2 - t1

    pbar = tqdm(
        iter(int, 1),
        desc=f"Evaluating {stage} set",
        total=len(eval_dataloader),
        disable=True,
    )

    eval_model.eval()
    device = torch.device("cpu")

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)
    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.
    two_filler_batches = itertools.islice(
        iter(eval_dataloader), TRAIN_PIPELINE_STAGES - 1
    )
    iterator = itertools.chain(iterator, two_filler_batches)

    preds = []
    labels = []

    auroc_computer = metrics.AUROC(task="binary").to(device)

    total_t = 0
    it = 0
    ctx1 = torch.no_grad()
    ctx2 = torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)
    with ctx1, ctx2:
        while True:
            try:
                logits, label, fw_t = eval_step(eval_model, iterator, it)
                if it > args.warmup_batches:
                    total_t += fw_t
                pred = torch.sigmoid(logits)
                preds.append(pred)
                labels.append(label)
                pbar.update(1)
                it += 1
            except StopIteration:
                # Dataset traversal complete
                break

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    num_samples = labels.shape[0] - args.warmup_batches * args.batch_size
    auroc = auroc = auroc_computer(preds.squeeze().float(), labels.float())
    print(f"AUROC over {stage} set: {auroc}.")
    print(f"Number of {stage} samples: {num_samples}")
    print(f"Throughput: {num_samples/total_t} fps")
    print(f"Final AUROC: {auroc} ")
    return auroc


def construct_model(args):
    device: torch.device = torch.device("cpu")
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    dcn_init_fn = OPTIMIZED_DLRM_DCN
    dlrm_model = dcn_init_fn(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("cpu")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dense_device=device,
    )

    train_model = DLRMTrain(dlrm_model)
    assert args.model_path
    load_snapshot(train_model, args.model_path)

    replace_crossnet(train_model.model)
    return train_model


def main(argv: List[str]) -> None:
    """
    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    backend = "gloo"
    pprint(vars(args))
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    model = construct_model(args)
    model.model.sparse_arch = model.model.sparse_arch.bfloat16()

    qconfig = FP8Config(
        fp8_config="E4M3",
        use_qdq=True,
        scale_method="MAXABS_ARBITRARY",
        dump_stats_path=args.out_dir,
    )

    if args.calib:
        test_dataloader = get_dataloader(args, backend, "test")
        model.model = prepare(model.model, qconfig)

        batch = fetch_batch(test_dataloader)
        batch.sparse_features = unpack(batch.sparse_features)
        batch_idx = list(range(128000))
        batch = test_dataloader.dataset.load_batch(batch_idx)
        batch.sparse_features = unpack(batch.sparse_features)
        model.model(batch.dense_features, batch.sparse_features)

        finalize_calibration(model.model)

    if args.quant:
        model.model = convert(model.model, qconfig)

    if args.accuracy:
        val_dataloader = get_dataloader(args, backend, "val")
        model = torch.compile(model)

        _evaluate(
        model.model,
        val_dataloader,
        "test",
        args,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
