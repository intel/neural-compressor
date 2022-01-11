import tensorflow as tf
from neural_compressor.experimental import Pruning
from neural_compressor.utils import logger

def generate_model(model_name='InceptionV3'):
    model = getattr(tf.keras.applications, model_name)(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
    )
    return model

def get_vgg16_baseline(model_path):
    if prune.train_distributed == True:
        import horovod.tensorflow as hvd
        hvd.init()
        if hvd.rank() == 0:
            model = generate_model('InceptionV3')
            model.summary()
            model.save(model_path)
    else:
        model = generate_model('InceptionV3')
        model.summary()
        model.save(model_path)
    return model_path


if __name__ == '__main__':
    prune = Pruning("./prune_inception_v3.yaml")
    # prune.train_distributed = True
    # prune.evaluation_distributed = True
    model_path = get_vgg16_baseline('./Inception-V3_Model')
    prune.model = model_path
    model = prune.fit()
    stats, sparsity = model.report_sparsity()
    logger.info(stats)
    logger.info(sparsity)
