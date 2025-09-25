"""Example: post-training quantization with neural-compressor.

Simplified version that keeps tf.flags usage for consistency
with the original script. Uses print for logging to stay minimal.
"""

from pathlib import Path
import tensorflow as tf

from neural_compressor import Metric
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import (
    BilinearImagenetTransform,
    ComposeTransform,
    DefaultDataLoader,
    TensorflowImageRecord,
)
from neural_compressor.quantization import fit

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# keep the same flag as original code
flags.DEFINE_string(
    "dataset_location",
    None,
    "location of calibration dataset and evaluate dataset",
)


def build_dataloader(root: str, batch_size: int) -> DefaultDataLoader:
    """Create a DefaultDataLoader for given root and batch size."""
    transform = ComposeTransform(
        transform_list=[BilinearImagenetTransform(height=224, width=224)]
    )
    dataset = TensorflowImageRecord(root=root, transform=transform)
    return DefaultDataLoader(dataset=dataset, batch_size=batch_size)


def main() -> None:
    """Run post-training quantization with predefined configuration."""
    dataset_location = FLAGS.dataset_location or "./dataset"
    model_path = "./mobilenet_v1_1.0_224_frozen.pb"
    calib_batch_size = 10
    eval_batch_size = 1
    calib_size = 20

    # basic checks
    ds_path = Path(dataset_location)
    model_file = Path(model_path)

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {ds_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # build dataloaders
    calib_dataloader = build_dataloader(root=str(ds_path), batch_size=calib_batch_size)
    eval_dataloader = build_dataloader(root=str(ds_path), batch_size=eval_batch_size)

    # metric and config
    top1 = Metric(name="topk", k=1)
    config = PostTrainingQuantConfig(calibration_sampling_size=[calib_size])

    q_model = fit(
        model=str(model_file),
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader,
        eval_metric=top1,
    )


if __name__ == "__main__":
    main()
