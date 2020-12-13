"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import io
import os
import time
import argparse
import random
import logging
import warnings
from functools import partial
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib.amp import amp
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer
from gluonnlp.model import BERTClassifier, RoBERTaClassifier
from gluonnlp.calibration import BertLayerCollector
from data.classification import MRPCTask, QQPTask, RTETask, STSBTask, SSTTask
from data.classification import QNLITask, CoLATask, MNLITask, WNLITask, XNLITask
from data.classification import LCQMCTask, ChnSentiCorpTask
from data.preprocessing_utils import truncate_seqs_equal, concat_sequences

nlp.utils.check_version('0.9', warning_only=True)

tasks = {
    'MRPC': MRPCTask(),
    'QQP': QQPTask(),
    'QNLI': QNLITask(),
    'RTE': RTETask(),
    'STS-B': STSBTask(),
    'CoLA': CoLATask(),
    'MNLI': MNLITask(),
    'WNLI': WNLITask(),
    'SST': SSTTask(),
    'XNLI': XNLITask(),
    'LCQMC': LCQMCTask(),
    'ChnSentiCorp': ChnSentiCorpTask()
}

parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--optimizer', type=str, default='bertadam',
                    help='The optimizer to be used for training')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs.')
parser.add_argument(
    '--training_steps', type=int, help='The total training steps. '
    'Note that if specified, epochs will be ignored.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size. Number of examples per gpu in a minibatch.')
parser.add_argument(
    '--dev_batch_size',
    type=int,
    default=8,
    help='Batch size for dev set and test set')
parser.add_argument(
    '--lr',
    type=float,
    default=3e-5,
    help='Initial learning rate')
parser.add_argument(
    '--epsilon',
    type=float,
    default=1e-6,
    help='Small value to avoid division by 0'
)
parser.add_argument(
    '--warmup_ratio',
    type=float,
    default=0.1,
    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='report interval')
parser.add_argument(
    '--max_len',
    type=int,
    default=128,
    help='Maximum length of the sentence pairs')
parser.add_argument(
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')
parser.add_argument(
    '--seed', type=int, default=2, help='Random seed')
parser.add_argument(
    '--accumulate',
    type=int,
    default=None,
    help='The number of batches for gradients accumulation to simulate large batch size. '
         'Default is None')
parser.add_argument(
    '--gpu', type=int, default=None, help='Which gpu for finetuning.')
parser.add_argument(
    '--task_name',
    type=str,
    choices=tasks.keys(),
    help='The name of the task to fine-tune. Choices include MRPC, QQP, '
         'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
parser.add_argument(
    '--bert_model',
    type=str,
    default='bert_12_768_12',
    choices=['bert_12_768_12', 'bert_24_1024_16', 'roberta_12_768_12', 'roberta_24_1024_16'],
    help='The name of pre-trained BERT model to fine-tune')
parser.add_argument(
    '--bert_dataset',
    type=str,
    default='book_corpus_wiki_en_uncased',
    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
             'openwebtext_book_corpus_wiki_en_uncased', 'wiki_multilingual_uncased',
             'wiki_multilingual_cased', 'wiki_cn_cased',
             'openwebtext_ccnews_stories_books_cased'],
    help='The dataset BERT pre-trained with.')
parser.add_argument(
    '--pretrained_bert_parameters',
    type=str,
    default=None,
    help='Pre-trained bert model parameter file.')
parser.add_argument(
    '--model_parameters',
    type=str,
    default=None,
    help='A parameter file for the model that is loaded into the model'
    ' before training/inference. It is different from the parameter'
    ' file written after the model is trained.')
parser.add_argument(
    '--output_dir',
    type=str,
    default='./output_dir',
    help='The output directory where the model params will be written.')
parser.add_argument(
    '--only_inference',
    action='store_true',
    help='If set, we skip training and only perform inference on dev and test data.')
parser.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float16'],
    help='The data type for training.')
parser.add_argument(
    '--early_stop',
    type=int,
    default=None,
    help='Whether to perform early stopping based on the metric on dev set. '
         'The provided value is the patience. ')
parser.add_argument('--deploy', action='store_true',
                    help='whether load static model for deployment')
parser.add_argument('--model_prefix', type=str, required=False,
                    help='load static model as hybridblock.')
parser.add_argument('--only_calibration', action='store_true',
                    help='quantize model')
parser.add_argument('--num_calib_batches', type=int, default=5,
                    help='number of batches for calibration')
parser.add_argument('--quantized_dtype', type=str, default='auto',
                    choices=['auto', 'int8', 'uint8'],
                    help='quantization destination data type for input data')
parser.add_argument('--calib_mode', type=str, default='customize',
                    choices=['none', 'naive', 'entropy', 'customize'],
                    help='calibration mode used for generating calibration table '
                         'for the quantized symbol.')
parser.add_argument('--tune',action='store_true', default=False,
                    help='Get bert tuning quantization model with lpot.')

args = parser.parse_args()


log = logging.getLogger()
log.setLevel(logging.INFO)

logging.captureWarnings(True)
fh = logging.FileHandler('log_{0}.txt'.format(args.task_name))
formatter = logging.Formatter(fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s',
                              datefmt='%H:%M:%S')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)
logging.info(args)

batch_size = args.batch_size
dev_batch_size = args.dev_batch_size
task_name = args.task_name
lr = args.lr
epsilon = args.epsilon
accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    logging.info('Using gradient accumulation. Effective batch size = ' \
                 'batch_size * accumulate = %d', accumulate * batch_size)

# random seed
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)

ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)

task = tasks[task_name]

# data type with mixed precision training
if args.dtype == 'float16':
    amp.init()

# model and loss
only_inference = args.only_inference
model_name = args.bert_model
dataset = args.bert_dataset
pretrained_bert_parameters = args.pretrained_bert_parameters
model_parameters = args.model_parameters

# load symbolic model
deploy = args.deploy
model_prefix = args.model_prefix

if only_inference and not model_parameters:
    warnings.warn('model_parameters is not set. '
                  'Randomly initialized model will be used for inference.')

get_pretrained = not (pretrained_bert_parameters is not None or model_parameters is not None)

use_roberta = 'roberta' in model_name
get_model_params = {
    'name': model_name,
    'dataset_name': dataset,
    'pretrained': get_pretrained,
    'ctx': ctx,
    'use_decoder': False,
    'use_classifier': False,
}
# RoBERTa does not contain parameters for sentence pair classification
if not use_roberta:
    get_model_params['use_pooler'] = True

bert, vocabulary = nlp.model.get_model(**get_model_params)

# initialize the rest of the parameters
initializer = mx.init.Normal(0.02)
# STS-B is a regression task.
# STSBTask().class_labels returns None
do_regression = not task.class_labels
if do_regression:
    num_classes = 1
    loss_function = gluon.loss.L2Loss()
else:
    num_classes = len(task.class_labels)
    loss_function = gluon.loss.SoftmaxCELoss()
# reuse the BERTClassifier class with num_classes=1 for regression
if use_roberta:
    model = RoBERTaClassifier(bert, dropout=0.0, num_classes=num_classes)
else:
    model = BERTClassifier(bert, dropout=0.1, num_classes=num_classes)
# initialize classifier
if not model_parameters:
    model.classifier.initialize(init=initializer, ctx=ctx)

# load checkpointing
output_dir = args.output_dir
if pretrained_bert_parameters:
    logging.info('loading bert params from %s', pretrained_bert_parameters)
    nlp.utils.load_parameters(model.bert, pretrained_bert_parameters, ctx=ctx, ignore_extra=True,
                              cast_dtype=True)
if model_parameters:
    logging.info('loading model params from %s', model_parameters)
    nlp.utils.load_parameters(model, model_parameters, ctx=ctx, cast_dtype=True)
nlp.utils.mkdir(output_dir)

logging.debug(model)
model.hybridize(static_alloc=True)
loss_function.hybridize(static_alloc=True)

if deploy:
    logging.info('load symbol file directly as SymbolBlock for model deployment')
    model = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(args.model_prefix),
                                         ['data0', 'data1', 'data2'],
                                         '{}-0000.params'.format(args.model_prefix))
    model.hybridize(static_alloc=True, static_shape=True)

# data processing
do_lower_case = 'uncased' in dataset
if use_roberta:
    bert_tokenizer = nlp.data.GPT2BPETokenizer()
else:
    bert_tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)

# calibration config
only_calibration = args.only_calibration
num_calib_batches = args.num_calib_batches
quantized_dtype = args.quantized_dtype
calib_mode = args.calib_mode

def convert_examples_to_features(example, tokenizer=None, truncate_length=512, cls_token=None,
                                 sep_token=None, class_labels=None, label_alias=None, vocab=None,
                                 is_test=False):
    """convert glue examples into necessary features"""
    if not is_test:
        label_dtype = 'int32' if class_labels else 'float32'
        # get the label
        label = example[-1]
        example = example[:-1]
        #create label maps if classification task
        if class_labels:
            label_map = {}
            for (i, l) in enumerate(class_labels):
                label_map[l] = i
            if label_alias:
                for key in label_alias:
                    label_map[key] = label_map[label_alias[key]]
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # tokenize raw text
    tokens_raw = [tokenizer(l) for l in example]
    # truncate to the truncate_length,
    tokens_trun = truncate_seqs_equal(tokens_raw, truncate_length)
    # concate the sequences with special tokens
    tokens_trun[0] = [cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = concat_sequences(tokens_trun, [[sep_token]] * len(tokens_trun))
    # convert the token to ids
    input_ids = vocab[tokens]
    valid_length = len(input_ids)
    if not is_test:
        return input_ids, segment_ids, valid_length, label
    else:
        return input_ids, segment_ids, valid_length


def preprocess_data(tokenizer, task, batch_size, dev_batch_size, max_len, vocab):
    """Train/eval Data preparation function."""
    label_dtype = 'int32' if task.class_labels else 'float32'
    truncate_length = max_len - 3 if task.is_pair else max_len - 2
    trans = partial(convert_examples_to_features, tokenizer=tokenizer,
                    truncate_length=truncate_length,
                    cls_token=vocab.cls_token if not use_roberta else vocab.bos_token,
                    sep_token=vocab.sep_token if not use_roberta else vocab.eos_token,
                    class_labels=task.class_labels, label_alias=task.label_alias, vocab=vocab)

    # data train
    # task.dataset_train returns (segment_name, dataset)
    train_tsv = task.dataset_train()[1]
    data_train = mx.gluon.data.SimpleDataset(list(map(trans, train_tsv)))
    data_train_len = data_train.transform(lambda _, segment_ids, valid_length, label: valid_length,
                                          lazy=False)
    # bucket sampler for training
    pad_val = vocabulary[vocabulary.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=args.round_to),  # input
        nlp.data.batchify.Pad(axis=0, pad_val=0, round_to=args.round_to),  # segment
        nlp.data.batchify.Stack(),  # length
        nlp.data.batchify.Stack(label_dtype))  # label
    batch_sampler = nlp.data.sampler.FixedBucketSampler(data_train_len, batch_size=batch_size,
                                                        num_buckets=10, ratio=0, shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(dataset=data_train, num_workers=4,
                                         batch_sampler=batch_sampler, batchify_fn=batchify_fn)

    # data dev. For MNLI, more than one dev set is available
    dev_tsv = task.dataset_dev()
    dev_tsv_list = dev_tsv if isinstance(dev_tsv, list) else [dev_tsv]
    loader_dev_list = []
    for segment, data in dev_tsv_list:
        data_dev = mx.gluon.data.SimpleDataset(list(map(trans, data)))
        loader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=dev_batch_size, num_workers=4,
                                              shuffle=False, batchify_fn=batchify_fn)
        loader_dev_list.append((segment, loader_dev))

    # batchify for data test
    test_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=pad_val, round_to=args.round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=0, round_to=args.round_to),
        nlp.data.batchify.Stack())
    # transform for data test
    test_trans = partial(convert_examples_to_features, tokenizer=tokenizer, truncate_length=max_len,
                         cls_token=vocab.cls_token if not use_roberta else vocab.bos_token,
                         sep_token=vocab.sep_token if not use_roberta else vocab.eos_token,
                         class_labels=None, is_test=True, vocab=vocab)

    # data test. For MNLI, more than one test set is available
    test_tsv = task.dataset_test()
    test_tsv_list = test_tsv if isinstance(test_tsv, list) else [test_tsv]
    loader_test_list = []
    for segment, data in test_tsv_list:
        data_test = mx.gluon.data.SimpleDataset(list(map(test_trans, data)))
        loader_test = mx.gluon.data.DataLoader(data_test, batch_size=dev_batch_size, num_workers=4,
                                               shuffle=False, batchify_fn=test_batchify_fn)
        loader_test_list.append((segment, loader_test))
    return loader_train, loader_dev_list, loader_test_list, len(data_train)


# Get the loader.
logging.info('processing dataset...')
train_data, dev_data_list, test_data_list, num_train_examples = preprocess_data(
    bert_tokenizer, task, batch_size, dev_batch_size, args.max_len, vocabulary)

def calibration(net, dev_data_list, num_calib_batches, quantized_dtype, calib_mode):
    """calibration function on the dev dataset."""
    assert len(dev_data_list) == 1, \
        'Currectly, MNLI not supported.'
    assert ctx == mx.cpu(), \
        'Currently only supports CPU with MKL-DNN backend.'
    logging.info('Now we are doing calibration on dev with %s.', ctx)
    for _, dev_data in dev_data_list:
        collector = BertLayerCollector(clip_min=-50, clip_max=10, logger=logging)
        num_calib_examples = dev_batch_size * num_calib_batches
        net = mx.contrib.quantization.quantize_net_v2(net, quantized_dtype=quantized_dtype,
                                                      exclude_layers=[],
                                                      quantize_mode='smart',
                                                      quantize_granularity='channel-wise',
                                                      calib_data=dev_data,
                                                      calib_mode=calib_mode,
                                                      num_calib_examples=num_calib_examples,
                                                      ctx=ctx,
                                                      LayerOutputCollector=collector,
                                                      logger=logging)
        # save params
        ckpt_name = 'model_bert_{0}_quantized_{1}'.format(task_name, calib_mode)
        params_saved = os.path.join(output_dir, ckpt_name)
        net.export(params_saved, epoch=0)
        logging.info('Saving quantized model at %s', output_dir)


def test(loader_test, segment):
    """Inference function on the test dataset."""
    logging.info('Now we are doing testing on %s with %s.', segment, ctx)

    tic = time.time()
    results = []
    for _, seqs in enumerate(loader_test):
        input_ids, segment_ids, valid_length = seqs
        input_ids = input_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx).astype('float32')
        if use_roberta:
            out = model(input_ids, valid_length)
        else:
            out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
        if not task.class_labels:
            # regression task
            for result in out.asnumpy().reshape(-1).tolist():
                results.append('{:.3f}'.format(result))
        else:
            # classification task
            indices = mx.nd.topk(out, k=1, ret_typ='indices', dtype='int32').asnumpy()
            for index in indices:
                results.append(task.class_labels[int(index)])

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_test) / (toc - tic))
    # write result to a file.
    segment = segment.replace('_mismatched', '-mm')
    segment = segment.replace('_matched', '-m')
    segment = segment.replace('SST', 'SST-2')
    filename = args.task_name + segment.replace('test', '') + '.tsv'
    test_path = os.path.join(args.output_dir, filename)
    with io.open(test_path, 'w', encoding='utf-8') as f:
        f.write(u'index\tprediction\n')
        for i, pred in enumerate(results):
            f.write(u'%d\t%s\n' % (i, str(pred)))


def log_train(batch_id, batch_num, metric, step_loss, log_interval, epoch_id, learning_rate):
    """Generate and print out the log message for training. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    train_str = '[Epoch %d Batch %d/%d] loss=%.4f, lr=%.7f, metrics:' + \
                ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(train_str, epoch_id + 1, batch_id + 1, batch_num, step_loss / log_interval,
                 learning_rate, *metric_val)


def log_eval(batch_id, batch_num, metric, step_loss, log_interval):
    """Generate and print out the log message for inference. """
    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]

    eval_str = '[Batch %d/%d] loss=%.4f, metrics:' + \
               ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(eval_str, batch_id + 1, batch_num, step_loss / log_interval, *metric_val)


def train(metric):
    """Training function."""
    if not only_inference:
        logging.info('Now we are doing BERT classification training on %s!', ctx)

    all_model_params = model.collect_params()
    optimizer_params = {'learning_rate': lr, 'epsilon': epsilon, 'wd': 0.01}
    trainer = gluon.Trainer(all_model_params, args.optimizer, optimizer_params,
                            update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    epoch_number = args.epochs
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    if args.training_steps:
        num_train_steps = args.training_steps
        epoch_number = 9999

    logging.info('training steps=%d', num_train_steps)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in all_model_params.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'
    # track best eval score
    metric_history = []
    best_metric = None
    patience = args.early_stop

    tic = time.time()
    finish_flag = False
    for epoch_id in range(epoch_number):
        if args.early_stop and patience == 0:
            logging.info('Early stopping at epoch %d', epoch_id)
            break
        if finish_flag:
            break
        if not only_inference:
            metric.reset()
            step_loss = 0
            tic = time.time()
            all_model_params.zero_grad()

            for batch_id, seqs in enumerate(train_data):
                # learning rate schedule
                if step_num < num_warmup_steps:
                    new_lr = lr * step_num / num_warmup_steps
                else:
                    non_warmup_steps = step_num - num_warmup_steps
                    offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                    new_lr = lr - offset * lr
                trainer.set_learning_rate(new_lr)

                # forward and backward
                with mx.autograd.record():
                    input_ids, segment_ids, valid_length, label = seqs
                    input_ids = input_ids.as_in_context(ctx)
                    valid_length = valid_length.as_in_context(ctx).astype('float32')
                    label = label.as_in_context(ctx)
                    if use_roberta:
                        out = model(input_ids, valid_length)
                    else:
                        out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)
                    ls = loss_function(out, label).mean()
                    if args.dtype == 'float16':
                        with amp.scale_loss(ls, trainer) as scaled_loss:
                            mx.autograd.backward(scaled_loss)
                    else:
                        ls.backward()

                # update
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(accumulate if accumulate else 1)
                    step_num += 1
                    if accumulate and accumulate > 1:
                        # set grad to zero for gradient accumulation
                        all_model_params.zero_grad()

                step_loss += ls.asscalar()
                if not do_regression:
                    label = label.reshape((-1))
                metric.update([label], [out])
                if (batch_id + 1) % (args.log_interval) == 0:
                    log_train(batch_id, len(train_data), metric, step_loss, args.log_interval,
                              epoch_id, trainer.learning_rate)
                    step_loss = 0
                if step_num >= num_train_steps:
                    logging.info('Finish training step: %d', step_num)
                    finish_flag = True
                    break
            mx.nd.waitall()

        # inference on dev data
        for segment, dev_data in dev_data_list:
            metric_nm, metric_val = evaluate(dev_data, metric, segment)
            if best_metric is None or metric_val >= best_metric:
                best_metric = metric_val
                patience = args.early_stop
            else:
                if args.early_stop is not None:
                    patience -= 1
            metric_history.append((epoch_id, metric_nm, metric_val))

        if not only_inference:
            # save params
            ckpt_name = 'model_bert_{0}_{1}.params'.format(task_name, epoch_id)
            params_saved = os.path.join(output_dir, ckpt_name)

            nlp.utils.save_parameters(model, params_saved)
            logging.info('params saved in: %s', params_saved)
            toc = time.time()
            logging.info('Time cost=%.2fs', toc - tic)
            tic = toc

    if not only_inference:
        # we choose the best model based on metric[0],
        # assuming higher score stands for better model quality
        metric_history.sort(key=lambda x: x[2][0], reverse=True)
        epoch_id, metric_nm, metric_val = metric_history[0]
        ckpt_name = 'model_bert_{0}_{1}.params'.format(task_name, epoch_id)
        params_saved = os.path.join(output_dir, ckpt_name)
        nlp.utils.load_parameters(model, params_saved)
        metric_str = 'Best model at epoch {}. Validation metrics:'.format(epoch_id)
        metric_str += ','.join([i + ':%.4f' for i in metric_nm])
        logging.info(metric_str, *metric_val)

    # inference on test data
    for segment, test_data in test_data_list:
        test(test_data, segment)


def evaluate(model, loader_dev, metric, segment):
    """Evaluate the model on validation dataset."""
    logging.info('Now we are doing evaluation on %s with %s.', segment, ctx)
    metric.reset()
    step_loss = 0
    tic = time.time()
    for batch_id, seqs in enumerate(loader_dev):
        input_ids, segment_ids, valid_length, label = seqs
        input_ids = input_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx).astype('float32')
        label = label.as_in_context(ctx)
        if use_roberta:
            out = model(input_ids, valid_length)
        else:
            out = model(input_ids, segment_ids.as_in_context(ctx), valid_length)

        ls = loss_function(out, label).mean()
        step_loss += ls.asscalar()
        if not do_regression:
            label = label.reshape((-1))
        metric.update([label], [out])
        if (batch_id + 1) % (args.log_interval) == 0:
            log_eval(batch_id, len(loader_dev), metric, step_loss, args.log_interval)
            step_loss = 0

    metric_nm, metric_val = metric.get()
    if not isinstance(metric_nm, list):
        metric_nm, metric_val = [metric_nm], [metric_val]
    metric_str = 'validation metrics:' + ','.join([i + ':%.4f' for i in metric_nm])
    logging.info(metric_str, *metric_val)

    mx.nd.waitall()
    toc = time.time()
    logging.info('Time cost=%.2fs, throughput=%.2f samples/s', toc - tic,
                 dev_batch_size * len(loader_dev) / (toc - tic))
    return metric_nm, metric_val

def test_func(graph):
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='[%H:%M:%S]',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MXNet-Demo")

    # dev_data_list=customized_args.customized_dict['dev_data_list']
    # metric=customized_args.customized_dict['metric']

    graph.hybridize(static_alloc=True, static_shape=True)
    tic = time.time()
    total_iter = 0
    for segment, dev_data in dev_data_list:
        metric_nm, metric_val = evaluate(model=graph, 
                                         loader_dev=dev_data, 
                                         metric=task.metrics, 
                                         segment=segment)
        total_iter += len(dev_data)
    toc = time.time()

    acc = metric_val[0]
    F1 = metric_val[1]
    speed = dev_batch_size * total_iter / (toc - tic)

    logger.info('Auto_tuning test DONE!')
    logger.info("%s is %.4f." % (metric_nm[0], acc))
    logger.info("%s is %.4f." % (metric_nm[1], F1))
    logger.info("speed is %.2f samples/s" % speed)

    return acc

if __name__ == '__main__':
    if only_calibration:
        try:
            calibration(model,
                        dev_data_list,
                        num_calib_batches,
                        quantized_dtype,
                        calib_mode)
        except AttributeError:
            nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
            warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')
    elif args.tune:
        # lpot auto-tuning
        if only_inference:
            calib_data = dev_data_list[0][1]
            from lpot import Quantization
            quantizer = Quantization("./bert.yaml")
            quantizer(model, q_dataloader=calib_data, eval_dataloader=calib_data, eval_func=test_func)
    else:
        train(task.metrics)
