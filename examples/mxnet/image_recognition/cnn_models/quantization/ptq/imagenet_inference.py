# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import argparse
import logging
import time

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from neural_compressor.adaptor.mxnet_utils.util import check_mx_version, get_backend_name

if check_mx_version('2.0.0') or not check_mx_version('1.7.0'):  # version >= 2.0.0 or == 1.6.0
    from mxnet.contrib.quantization import quantize_net
else:
    from mxnet.contrib.quantization import quantize_net_v2 as quantize_net

mx.npx.reset_np()


class DataiterWrapper(mx.gluon.data.DataLoader):
    def __init__(self, dataiter, batch_size, discard_label=False):
        super().__init__(mx.gluon.data.SimpleDataset([]), batch_size)
        self.dataiter = dataiter
        self.batch_size = batch_size
        self.discard_label = discard_label

    def __iter__(self):
        self.dataiter.reset()
        for batch in self.dataiter:
            if self.discard_label:
                yield batch.data
            yield batch.data + batch.label

    def __bool__(self):
        return bool(self.dataiter.iter_next())

    def without_label(self):  # workaround for an MXNet 1.6 bug in the quantize_net function
        return DataiterWrapper(self.dataiter, self.batch_size, True)


def quantize(net, ctx, dataloader, batch_size, num_calib_batches, save_path, calib_mode, logger):
    # quantize and tune
    if calib_mode in ['naive', 'entropy']:
        calib_samples = {}
        if check_mx_version('2.0.0'):
            calib_samples['num_calib_batches'] = num_calib_batches
        else:
            calib_samples['num_calib_examples'] = num_calib_batches*batch_size
        qnet = quantize_net(net, ctx=ctx, calib_mode=calib_mode,
                            calib_data=dataloader.without_label(),
                            quantized_dtype='auto', logger=logger, **calib_samples)
    elif calib_mode == 'inc':
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization("./cnn.yaml")
        quantizer.model = common.Model(net)
        quantizer.calib_dataloader = dataloader
        quantizer.eval_dataloader = dataloader
        qnet = quantizer.fit().model
    else:
        raise ValueError(
            'Unknow calibration mode {} received, only supports `naive`, '
            '`entropy`, and `inc`'.format(calib_mode))

    data = next(iter(dataloader))[0].as_in_context(ctx)
    if check_mx_version('1.7.0'):
        qnet.optimize_for(data, backend=get_backend_name(ctx), static_alloc=True, static_shape=True)
    qnet.export(save_path, 0)
    logger.info('Saved quantized model to: {}'.format(save_path))

    return qnet


def score(symblock, ctx, data, max_num_examples, logger=None):
    if check_mx_version('2.0.0'):
        metrics = [gluon.metric.create('acc'),
                   gluon.metric.create('top_k_accuracy', top_k=5)]
    else:
        metrics = [mx.metric.create('acc'),
                   mx.metric.create('top_k_accuracy', top_k=5)]

    tic = time.time()
    num = 0
    for input_data in data:
        x = input_data[0].as_in_context(ctx)
        label = input_data[1].as_in_context(ctx)
        outputs = symblock.forward(x)
        for m in metrics:
            m.update(label, outputs)
        num += x.shape[0]  # batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    speed = num / (time.time() - tic)

    if logger is not None:
        logger.info('Finished inference with %d images' % num)
        logger.info('Finished with %f images per second', speed)
        for m in metrics:
            logger.info(m.get())
    print("Accuracy: %.5f" % metrics[0].get()[1])


def benchmark_score(symblock, ctx, data_shape, num_batches):
    data = mx.random.uniform(-1.0, 1.0, shape=data_shape, ctx=ctx)
    dry_run = 5  # use 5 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        outs = symblock.forward(data)
        for output in outs:
            output.wait_to_read()

    return num_batches*data_shape[0]/(time.time() - tic)


def main():
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='cpu')  # currently unused
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=True, help='param file path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path', default=None)
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60,
                        help='number of threads for data decoding')
    parser.add_argument('--shuffle-dataset', action='store_true', default=False,
                        help='shuffle the dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-inference-batches', type=int, required=True,
                        help='number of images used for inference')
    parser.add_argument("--output-graph",
                        help='Specify tune result model save dir',
                        dest='output_graph')
    parser.add_argument('--calib-mode', type=str, default='inc',
                        help="""Possible values:
                                    "naive": take min and max values of layer outputs as thresholds.
                                    "entropy": minimize the KL divergence between the f32 and quantized
                                            output.
                                    "inc": use Intel Neural Compressor tuning
                                """)
    parser.add_argument('--benchmark', action='store_true', help='dummy data benchmark')
    parser.add_argument('--accuracy-only', action='store_true', help='accuracy only benchmark')

    args = parser.parse_args()
    calib_mode = args.calib_mode.lower()

    assert args.ctx == 'cpu', 'Currently only cpu is supported'
    ctx = mx.cpu()

    symbol_file = args.symbol_file
    param_file = args.param_file
    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    symnet = mx.sym.load(symbol_file)
    if check_mx_version('2.0.0'):
        symnet = symnet.optimize_for('ONEDNN')
    else:
        symnet = symnet.get_backend_symbol('MKLDNN')

    symblock = gluon.SymbolBlock(symnet, [mx.sym.var('data')])
    try:
        symblock.load_parameters(param_file, cast_dtype=True, dtype_source='saved')
    except AssertionError:
        symblock.load_parameters(param_file, cast_dtype=True, dtype_source='saved',
                                 allow_missing=True)
    params = symblock.collect_params()
    if 'softmax_label' in params:
        params['softmax_label'].shape = (batch_size,)
        params['softmax_label'].initialize(force_reinit=True)
    for param in params.values():
        param.grad_req = 'null'
    symblock.hybridize(static_alloc=True, static_shape=True)

    rgb_mean = args.rgb_mean
    rgb_std = args.rgb_std
    logger.info('rgb_mean = %s' % rgb_mean)
    logger.info('rgb_std = %s' % rgb_std)
    rgb_mean = {'mean_' + c: float(i) for c, i in zip(['r', 'g', 'b'], rgb_mean.split(','))}
    rgb_std = {'std_' + c: float(i) for c, i in zip(['r', 'g', 'b'], rgb_std.split(','))}

    combine_mean_std = {}
    combine_mean_std.update(rgb_mean)
    combine_mean_std.update(rgb_std)

    image_shape = args.image_shape
    data_shape = [int(i) for i in image_shape.split(',')]
    logger.info('Input data shape = %s' % str(data_shape))

    if args.benchmark:
        logger.info('Running model %s for inference' % symbol_file)
        speed = benchmark_score(symblock, ctx, [batch_size] +
                                data_shape, args.num_inference_batches)
        logger.info('batch size %2d, image/sec: %f', batch_size, speed)
        print('Latency: %.3f ms' % (1 / speed * 1000))
        print('Throughput: %.3f images/sec' % (speed))
        sys.exit()

    assert os.path.isfile(args.dataset)
    dataset_path = args.dataset
    logger.info('Dataset for inference: %s' % dataset_path)

    data_nthreads = args.data_nthreads
    dataloader = DataiterWrapper(mx.io.ImageRecordIter(
        path_imgrec=dataset_path,
        label_width=1,
        preprocess_threads=data_nthreads,
        batch_size=batch_size,
        data_shape=data_shape,
        label_name='softmax_label',
        rand_crop=False,
        rand_mirror=False,
        shuffle=args.shuffle_dataset,
        dtype='float32',
        ctx=args.ctx,
        **combine_mean_std), batch_size)

    if args.accuracy_only:
        score(symblock, ctx, dataloader, None, logger)
        sys.exit()

    # quantization & tuning
    symblock = quantize(symblock, ctx, dataloader, batch_size,
                        10, args.output_graph, calib_mode, logger)

    logger.info('Running tuned model for inference')
    num_inference_images = args.num_inference_batches * batch_size
    score(symblock, ctx, dataloader, num_inference_images, logger)


if __name__ == '__main__':
    main()
