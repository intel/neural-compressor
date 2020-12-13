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
"""
SQuAD with Bidirectional Encoder Representations from Transformers
==================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

import argparse
import collections
import json
import logging
import os
import io
import random
import time
import warnings
import itertools
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from gluonnlp.calibration import BertLayerCollector
from model.qa import BertForQALoss, BertForQA
from bert_qa_evaluate import get_F1_EM, predict, PredResult
from data.preprocessing_utils import improve_answer_span, \
    concat_sequences, tokenize_and_align_positions, get_doc_spans, align_position2doc_spans, \
    check_is_max_context, convert_squad_examples

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(
    description='BERT QA example.'
    'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')
parser.add_argument('--training_steps',
                    type=int,
                    help='training steps, epochs will be ignored '
                    'if trainin_steps is specified.')
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                    '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument(
    '--round_to', type=int, default=None,
    help='The length of padded sequences will be rounded up to be multiple of this argument.'
         'When round to is set to 8, training throughput may increase for mixed precision'
         'training on GPUs with tensorcores.')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu',
                    action='store_true',
                    help='use GPU instead of CPU')

parser.add_argument('--sentencepiece',
                    type=str,
                    default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

parser.add_argument('--debug',
                    action='store_true',
                    help='Run the example in test mode for sanity checks')

parser.add_argument('--dtype',
                    type=str,
                    default='float32',
                    help='Data type used for training. Either float32 or float16')

parser.add_argument('--comm_backend',
                    type=str,
                    default=None,
                    help='Communication backend. Set to horovod if horovod is used for '
                         'multi-GPU training')

parser.add_argument('--deploy', action='store_true',
                    help='whether load static model for deployment')

parser.add_argument('--model_prefix', type=str, required=False,
                    help='load static model as hybridblock.')

parser.add_argument('--only_calibration', action='store_true',
                    help='quantize model')

parser.add_argument('--num_calib_batches', type=int, default=10,
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

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(args.output_dir, 'finetune_squad.log'),
                         mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

if args.comm_backend == 'horovod':
    import horovod.mxnet as hvd
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    local_rank = hvd.local_rank()
else:
    rank = 0
    size = 1
    local_rank = 0

if args.dtype == 'float16':
    from mxnet.contrib import amp
    amp.init()

model_name = args.bert_model
dataset_name = args.bert_dataset
only_predict = args.only_predict
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
if pretrained_bert_parameters and model_parameters:
    raise ValueError('Cannot provide both pre-trained BERT parameters and '
                     'BertForQA model parameters.')
lower = args.uncased

batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.gpu(local_rank) if args.gpu else mx.cpu()

accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    log.info('Using gradient accumulation. Effective total batch size = {}'.
             format(accumulate*batch_size*size))

optimizer = args.optimizer
warmup_ratio = args.warmup_ratio


version_2 = args.version_2
null_score_diff_threshold = args.null_score_diff_threshold

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

# vocabulary and tokenizer
if args.sentencepiece:
    logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
    if dataset_name:
        warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                      'The vocabulary will be loaded based on --sentencepiece.')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
    dataset_name = None
else:
    vocab = None

pretrained = not model_parameters and not pretrained_bert_parameters and not args.sentencepiece
bert, vocab = nlp.model.get_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=vocab,
    pretrained=pretrained,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False)

if args.sentencepiece:
    tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
else:
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

# load symbolic model
deploy = args.deploy
model_prefix = args.model_prefix

net = BertForQA(bert=bert)
if model_parameters:
    # load complete BertForQA parameters
    nlp.utils.load_parameters(net, model_parameters, ctx=ctx, cast_dtype=True)
elif pretrained_bert_parameters:
    # only load BertModel parameters
    nlp.utils.load_parameters(bert, pretrained_bert_parameters, ctx=ctx,
                              ignore_extra=True, cast_dtype=True)
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
elif pretrained:
    # only load BertModel parameters
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
else:
    # no checkpoint is loaded
    net.initialize(init=mx.init.Normal(0.02), ctx=ctx)

net.hybridize(static_alloc=True)

loss_function = BertForQALoss()
loss_function.hybridize(static_alloc=True)

if deploy:
    logging.info('load symbol file directly as SymbolBlock for model deployment')
    net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(args.model_prefix),
                                       ['data0', 'data1', 'data2'],
                                       '{}-0000.params'.format(args.model_prefix))
    net.hybridize(static_alloc=True, static_shape=True)

# calibration config
only_calibration = args.only_calibration
num_calib_batches = args.num_calib_batches
quantized_dtype = args.quantized_dtype
calib_mode = args.calib_mode

def train():
    """Training function."""
    segment = 'train'  #if not args.debug else 'dev'
    log.info('Loading %s data...', segment)
    if version_2:
        train_data = SQuAD(segment, version='2.0')
    else:
        train_data = SQuAD(segment, version='1.1')
    if args.debug:
        sampled_data = [train_data[i] for i in range(0, 10000)]
        train_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in Train data:{}'.format(len(train_data)))
    train_data_transform = preprocess_dataset(
        tokenizer,
        train_data,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        input_features=True)

    log.info('The number of examples after preprocessing:{}'.format(
        len(train_data_transform)))

    sampler = nlp.data.SplitSampler(len(train_data_transform), num_parts=size,
                                    part_index=rank, even_size=True)
    num_train_examples = len(sampler)
    train_dataloader = mx.gluon.data.DataLoader(train_data_transform,
                                                batchify_fn=batchify_fn,
                                                batch_size=batch_size,
                                                num_workers=4,
                                                sampler=sampler)

    log.info('Start Training')

    optimizer_params = {'learning_rate': lr}
    param_dict = net.collect_params()
    if args.comm_backend == 'horovod':
        trainer = hvd.DistributedTrainer(param_dict, optimizer, optimizer_params)
    else:
        trainer = mx.gluon.Trainer(param_dict, optimizer, optimizer_params,
                                   update_on_kvstore=False)
    if args.dtype == 'float16':
        amp.init_trainer(trainer)

    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * args.epochs)
    if args.training_steps:
        num_train_steps = args.training_steps

    num_warmup_steps = int(num_train_steps * warmup_ratio)

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in param_dict.values() if p.grad_req != 'null']

    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'
    net.collect_params().zero_grad()

    epoch_tic = time.time()

    total_num = 0
    log_num = 0
    batch_id = 0
    step_loss = 0.0
    tic = time.time()
    step_num = 0

    tic = time.time()
    while step_num < num_train_steps:
        for _, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            _, inputs, token_types, valid_length, start_label, end_label = data
            num_labels = len(inputs)
            log_num += num_labels
            total_num += num_labels

            with mx.autograd.record():
                out = net(inputs.as_in_context(ctx),
                          token_types.as_in_context(ctx),
                          valid_length.as_in_context(ctx).astype('float32'))

                loss = loss_function(out, [
                    start_label.as_in_context(ctx).astype('float32'),
                    end_label.as_in_context(ctx).astype('float32')
                ]).sum() / num_labels

                if accumulate:
                    loss = loss / accumulate
                if args.dtype == 'float16':
                    with amp.scale_loss(loss, trainer) as l:
                        mx.autograd.backward(l)
                        norm_clip = 1.0 * size * trainer._amp_loss_scaler.loss_scale
                else:
                    mx.autograd.backward(loss)
                    norm_clip = 1.0 * size

            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, norm_clip)
                trainer.update(1)
                if accumulate:
                    param_dict.zero_grad()

            if args.comm_backend == 'horovod':
                step_loss += hvd.allreduce(loss, average=True).asscalar()
            else:
                step_loss += loss.asscalar()

            if (batch_id + 1) % log_interval == 0:
                toc = time.time()
                log.info('Batch: {}/{}, Loss={:.4f}, lr={:.7f} '
                         'Thoughput={:.2f} samples/s'
                         .format(batch_id % len(train_dataloader),
                                 len(train_dataloader), step_loss / log_interval,
                                 trainer.learning_rate, log_num/(toc - tic)))
                tic = time.time()
                step_loss = 0.0
                log_num = 0

            if step_num >= num_train_steps:
                break
            batch_id += 1

        log.info('Finish training step: %d', step_num)
        epoch_toc = time.time()
        log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
            epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))

    if rank == 0:
        net.save_parameters(os.path.join(output_dir, 'net.params'))

def calibration(net, num_calib_batches, quantized_dtype, calib_mode):
    """calibration function on the dev dataset."""
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    batchify_fn_calib = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))

    dev_data_transform = preprocess_dataset(tokenizer,
                                            dev_data,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            input_features=True,
                                            for_calibration=True)

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn_calib,
        num_workers=4, batch_size=test_batch_size,
        shuffle=False, last_batch='keep')

    assert ctx == mx.cpu(), \
        'Currently only supports CPU with MKL-DNN backend.'
    log.info('Now we are doing calibration on dev with %s.', ctx)
    collector = BertLayerCollector(clip_min=-50, clip_max=10, logger=log)
    num_calib_examples = test_batch_size * num_calib_batches
    net = mx.contrib.quantization.quantize_net_v2(net, quantized_dtype=quantized_dtype,
                                                  exclude_layers=[],
                                                  quantize_mode='smart',
                                                  quantize_granularity='channel-wise',
                                                  calib_data=dev_dataloader,
                                                  calib_mode=calib_mode,
                                                  num_calib_examples=num_calib_examples,
                                                  ctx=ctx,
                                                  LayerOutputCollector=collector,
                                                  logger=log)
    # save params
    ckpt_name = 'model_bert_squad_quantized_{0}'.format(calib_mode)
    params_saved = os.path.join(output_dir, ckpt_name)
    net.export(params_saved, epoch=0)
    log.info('Saving quantized model at %s', output_dir)

def evaluate(net):
    """Evaluate the model on validation dataset."""
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[i] for i in range(100)]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    dev_dataset = preprocess_dataset(tokenizer,
                                     dev_data,
                                     max_seq_length=max_seq_length,
                                     doc_stride=doc_stride,
                                     max_query_length=max_query_length,
                                     input_features=False)

    dev_data_transform = preprocess_dataset(tokenizer,
                                            dev_data,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            input_features=True)

    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(dev_data_transform,
                                              batchify_fn=batchify_fn,
                                              num_workers=4,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              last_batch='keep')

    log.info('start prediction')

    all_results = collections.defaultdict(list)

    epoch_tic = time.time()
    total_num = 0
    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        total_num += len(inputs)
        out = net(inputs.as_in_context(ctx),
                  token_types.as_in_context(ctx),
                  valid_length.as_in_context(ctx).astype('float32'))

        output = mx.nd.split(out, axis=2, num_outputs=2)
        example_ids = example_ids.asnumpy().tolist()
        pred_start = output[0].reshape((0, -3)).asnumpy()
        pred_end = output[1].reshape((0, -3)).asnumpy()

        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results[example_id].append(PredResult(start=start, end=end))

    epoch_toc = time.time()
    log.info('Time cost={:.2f} s, Thoughput={:.2f} samples/s'.format(
        epoch_toc - epoch_tic, total_num / (epoch_toc - epoch_tic)))

    log.info('Get prediction results...')

    all_predictions = collections.OrderedDict()

    for features in dev_dataset:
        results = all_results[features[0].example_id]
        example_qas_id = features[0].qas_id

        prediction, _ = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            n_best_size=n_best_size,
            version_2=version_2)

        all_predictions[example_qas_id] = prediction

    if version_2:
        log.info('Please run evaluate-v2.0.py to get evaluation results for SQuAD 2.0')
    else:
        F1_EM = get_F1_EM(dev_data, all_predictions)
        log.info(F1_EM)

    with io.open(os.path.join(output_dir, 'predictions.json'),
                 'w', encoding='utf-8') as fout:
        data = json.dumps(all_predictions, ensure_ascii=False)
        fout.write(data)

    return F1_EM["exact_match"]


SquadBERTFeautre = collections.namedtuple('SquadBERTFeautre', [
    'example_id', 'qas_id', 'doc_tokens', 'valid_length', 'tokens',
    'token_to_orig_map', 'token_is_max_context', 'input_ids', 'p_mask',
    'segment_ids', 'start_position', 'end_position', 'is_impossible'
])


def convert_examples_to_features(example,
                                 tokenizer=None,
                                 cls_token=None,
                                 sep_token=None,
                                 vocab=None,
                                 max_seq_length=384,
                                 doc_stride=128,
                                 max_query_length=64,
                                 cls_index=0):
    """convert the examples to the BERT features"""
    query_tokenized = [cls_token] + tokenizer(
        example.question_text)[:max_query_length]
    #tokenize paragraph and get start/end position of the answer in tokenized paragraph
    tok_start_position, tok_end_position, all_doc_tokens, _, tok_to_orig_index = \
        tokenize_and_align_positions(example.doc_tokens,
                                     example.start_position,
                                     example.end_position,
                                     tokenizer)
    # get doc spans using sliding window
    doc_spans, doc_spans_indices = get_doc_spans(
        all_doc_tokens, max_seq_length - len(query_tokenized) - 2, doc_stride)

    if not example.is_impossible:
        (tok_start_position, tok_end_position) = improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
        # get the new start/end position
        positions = [
            align_position2doc_spans([tok_start_position, tok_end_position],
                                     doc_idx,
                                     offset=len(query_tokenized) + 1,
                                     default_value=0)
            for doc_idx in doc_spans_indices
        ]
    else:
        # if the question is impossible to answer, set the start/end position to cls index
        positions = [[cls_index, cls_index] for _ in doc_spans_indices]

    # record whether the tokens in a docspan have max context
    token_is_max_context = [{
        len(query_tokenized) + p:
        check_is_max_context(doc_spans_indices, i, p + doc_spans_indices[i][0])
        for p in range(len(doc_span))
    } for (i, doc_span) in enumerate(doc_spans)]

    token_to_orig_map = [{
        len(query_tokenized) + p + 1:
        tok_to_orig_index[p + doc_spans_indices[i][0]]
        for p in range(len(doc_span))
    } for (i, doc_span) in enumerate(doc_spans)]

    #get sequence features: tokens, segment_ids, p_masks
    seq_features = [
        concat_sequences([query_tokenized, doc_span], [[sep_token]] * 2)
        for doc_span in doc_spans
    ]

    features = [
        SquadBERTFeautre(example_id=example.example_id,
                         qas_id=example.qas_id,
                         doc_tokens=example.doc_tokens,
                         valid_length=len(tokens),
                         tokens=tokens,
                         token_to_orig_map=t2o,
                         token_is_max_context=is_max,
                         input_ids=vocab[tokens],
                         p_mask=p_mask,
                         segment_ids=segment_ids,
                         start_position=start,
                         end_position=end,
                         is_impossible=example.is_impossible)
        for (tokens, segment_ids, p_mask), (start, end), is_max, t2o in zip(
            seq_features, positions, token_is_max_context, token_to_orig_map)
    ]
    return features


def preprocess_dataset(tokenizer,
                       dataset,
                       vocab=None,
                       max_seq_length=384,
                       doc_stride=128,
                       max_query_length=64,
                       input_features=True,
                       num_workers=4,
                       load_from_pickle=False,
                       feature_file=None,
                       for_calibration=False):
    """Loads a dataset into features"""
    vocab = tokenizer.vocab if vocab is None else vocab
    trans = partial(convert_examples_to_features,
                    tokenizer=tokenizer,
                    cls_token=vocab.cls_token,
                    sep_token=vocab.sep_token,
                    vocab=vocab,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_query_length=max_query_length)
    pool = mp.Pool(num_workers)
    start = time.time()
    if not load_from_pickle:
        example_trans = partial(convert_squad_examples,
                                is_training=input_features)
        # convert the raw dataset into raw features
        examples = pool.map(example_trans, dataset)
        raw_features = pool.map(trans, examples)
        if feature_file:
            with open(feature_file, 'wb') as file:
                pickle.dump(list(raw_features), file)
    else:
        assert feature_file, 'feature file should be provided.'
        with open(feature_file, 'wb') as file:
            raw_features = pickle.load(file)

    if input_features:
        # convert the full features into the training features
        # Note that we will need the full features to make evaluation
        # Due to using sliding windows in data preprocessing,
        # we will have multiple examples for a single entry after processed.
        # Thus we need to flatten it for training.
        data_feature = mx.gluon.data.SimpleDataset(
            list(itertools.chain.from_iterable(raw_features)))
        if for_calibration:
            data_feature = data_feature.transform(lambda *example: (
                example[7],  # inputs_id
                example[9],  # segment_ids
                example[3],  # valid_length,
                example[10]))  # start_position,
        else:
            data_feature = data_feature.transform(lambda *example: (
                example[0],  # example_id
                example[7],  # inputs_id
                example[9],  # segment_ids
                example[3],  # valid_length,
                example[10],  # start_position,
                example[11]))  # end_position
    else:
        data_feature = mx.gluon.data.SimpleDataset(list(raw_features))

    end = time.time()
    pool.close()
    print('Done! Transform dataset costs %.2f seconds.' % (end - start))
    return data_feature


def gen_dataset():
    """generate dataset for lpot."""
    log.info('Loading dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    if args.debug:
        sampled_data = [dev_data[0], dev_data[1], dev_data[2]]
        dev_data = mx.gluon.data.SimpleDataset(sampled_data)
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    batchify_fn_calib = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token], round_to=args.round_to),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))

    dev_data_transform = preprocess_dataset(tokenizer,
                                            dev_data,
                                            max_seq_length=max_seq_length,
                                            doc_stride=doc_stride,
                                            max_query_length=max_query_length,
                                            input_features=True,
                                            for_calibration=True)

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn_calib,
        num_workers=4, batch_size=test_batch_size,
        shuffle=False, last_batch='keep')
    
    return dev_dataloader

def eval_func(model):
    """evaluation function for lpot."""
    EM_acc = evaluate(model)
    return EM_acc

if __name__ == '__main__':
    if only_calibration:
        try:
            calibration(net,
                        num_calib_batches,
                        quantized_dtype,
                        calib_mode)
        except AttributeError:
            nlp.utils.version.check_version('1.7.0', warning_only=True, library=mx)
            warnings.warn('INT8 Quantization for BERT need mxnet-mkl >= 1.6.0b20200115')
    elif not only_predict:
        train()
        evaluate()
    elif args.tune:
        # lpot auto-tuning
        dev_dataloader = gen_dataset()
        from lpot import Quantization
        quantizer = Quantization("./bert.yaml")
        quantizer(net, q_dataloader=dev_dataloader, eval_dataloader=dev_dataloader, eval_func=eval_func)
    elif model_parameters or deploy:
        evaluate()

