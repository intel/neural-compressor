import tensorflow as tf

from neural_compressor.data import TensorflowImageRecord
from neural_compressor.data import BilinearImagenetTransform
from neural_compressor.data import ComposeTransform
from neural_compressor.data import DefaultDataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_location', None, 'location of calibration dataset and evaluate dataset')

calib_dataset = TensorflowImageRecord(root=FLAGS.dataset_location, transform= \
        ComposeTransform(transform_list= [BilinearImagenetTransform(height=224, width=224)]))
calib_dataloader = DefaultDataLoader(dataset=calib_dataset, batch_size=10)

eval_dataset = TensorflowImageRecord(root=FLAGS.dataset_location, transform=ComposeTransform(transform_list= \
        [BilinearImagenetTransform(height=224, width=224)]))
eval_dataloader = DefaultDataLoader(dataset=eval_dataset, batch_size=1)

def main():
    config = PostTrainingQuantConfig(calibration_sampling_size=[20])
    q_model = fit(
        model="./mobilenet_v1_1.0_224_frozen.pb",
        conf=config,
        calib_dataloader=calib_dataloader,
        eval_dataloader=eval_dataloader)

if __name__ == "__main__":
    main()
