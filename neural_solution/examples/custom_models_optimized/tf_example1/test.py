# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Running script."""
import tensorflow as tf

from neural_compressor import Metric
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from neural_compressor.data import BilinearImagenetTransform, ComposeTransform, DefaultDataLoader, TensorflowImageRecord
from neural_compressor.quantization import fit

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_location", None, "location of calibration dataset and evaluate dataset")

flags.DEFINE_string("model_path", None, "location of model")

calib_dataset = TensorflowImageRecord(
    root=FLAGS.dataset_location,
    transform=ComposeTransform(transform_list=[BilinearImagenetTransform(height=224, width=224)]),
)
calib_dataloader = DefaultDataLoader(dataset=calib_dataset, batch_size=10)

eval_dataset = TensorflowImageRecord(
    root=FLAGS.dataset_location,
    transform=ComposeTransform(transform_list=[BilinearImagenetTransform(height=224, width=224)]),
)
eval_dataloader = DefaultDataLoader(dataset=eval_dataset, batch_size=1)


def main():
    """Implement running function."""
    top1 = Metric(name="topk", k=1)
    tuning_criterion = TuningCriterion(strategy="basic")
    config = PostTrainingQuantConfig(calibration_sampling_size=[20], quant_level=1, tuning_criterion=tuning_criterion)
    model_path = FLAGS.model_path + "/mobilenet_v1_1.0_224_frozen.pb"
    q_model = fit(
        model=model_path,
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader,
        eval_metric=top1,
    )
    q_model.save("./q_model_path/q_model")


if __name__ == "__main__":
    main()
