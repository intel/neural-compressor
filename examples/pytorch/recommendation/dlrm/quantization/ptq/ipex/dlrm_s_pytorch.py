# Copyright (c) 2021 Intel Corporation
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
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import sys
import time


# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler

# intel
import intel_extension_for_pytorch as ipex
from torch.utils import ThroughputBenchmark
# For distributed run
import extend_distributed as ext_dist

try:
    from intel_extension_for_pytorch.quantization import prepare, convert
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    IPEX_112 = True
except:
    IPEX_112 = False


exc = getattr(builtins, "IOError", "FileNotFoundError")

def freeze(model):
    return torch.jit._recursive.wrap_cpp_module(torch._C._freeze_module(model._c, preserveParameters=True))


def time_wrap():
    return time.time()


def dlrm_wrap(X, *emb_args):
    with record_function("DLRM forward"):
        return dlrm(X, *emb_args)


def loss_fn_wrap(Z, T):
    with record_function("DLRM loss compute"):
        return dlrm.loss_fn(Z, T)

# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, local_ln_emb=None):
        emb_l = nn.ModuleList()
        n_embs = ln.size if local_ln_emb is None else len(local_ln_emb)
        for i in range(n_embs):
            if local_ln_emb is None:
                n = ln[i]
            else:
                n = ln[local_ln_emb[i]]
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            if not args.inference_only:
                nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            emb_l.append(EE)
        return emb_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        weighted_pooling=None,
        loss_threshold=0.0,
    ):
        super(DLRM_Net, self).__init__()
        self.loss_threshold = loss_threshold
        #If running distributed, get local slice of embedding tables
        if ext_dist.my_size > 1:
            n_emb = len(ln_emb)
            self.n_global_emb = n_emb
            self.rank = ext_dist.dist.get_rank()
            self.ln_emb = [i for i in range(n_emb)]
            self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(n_emb)
            self.local_ln_emb_slice = ext_dist.get_my_slice(n_emb)
            self.local_ln_emb = self.ln_emb[self.local_ln_emb_slice]
        else:
            self.local_ln_emb = None
        self.emb_l = self.create_emb(m_spa, ln_emb, self.local_ln_emb)
        self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
        self.top_l = self.create_mlp(ln_top, sigmoid_top)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")


    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, emb_l, *emb_args):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        if isinstance(emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
            return emb_l(emb_args, self.need_linearize_indices_and_offsets)
        lS_o, lS_i = emb_args
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
            )

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        if args.ipex_interaction:
            T = [x] + list(ly)
            R = ipex.nn.functional.interaction(*T)
        else:
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        return R

    def forward(self, dense_x, *emb_args):
        if ext_dist.my_size > 1:
            return self.distributed_forward(dense_x, *emb_args)
        else:
            return self.sequential_forward(dense_x, *emb_args)

    def distributed_forward(self, dense_x, *emb_args):
        batch_size = dense_x.size()[0]
        vector_lenght = self.emb_l.weights[0].size()[1]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < ext_dist.my_size:
            sys.exit("ERROR: batch_size (%d) must be larger than number of ranks (%d)" % (batch_size, ext_dist.my_size))

        # embeddings
        ly = self.apply_emb(self.emb_l, *emb_args)
        a2a_req = ext_dist.alltoall(ly, self.n_emb_per_rank)
        # bottom mlp
        x = self.apply_mlp(dense_x, self.bot_l)
        ly = a2a_req.wait()
        _ly = []
        for item in ly:
            _ly += [item[:, emb_id * vector_lenght: (emb_id + 1) * vector_lenght] for emb_id in range(self.emb_l.n_tables)]
        # interactions
        z = self.interact_features(x, _ly)
        # top mlp
        p = self.apply_mlp(z, self.top_l)
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(
                p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z = p
        return z
 

    def sequential_forward(self, dense_x, *emb_args):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(self.emb_l, *emb_args)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def trace_model(args, dlrm, test_ld, inplace=True):
    dlrm.eval()
    for j, inputBatch in enumerate(test_ld):
        X, lS_o, lS_i, _, _, _ = unpack_batch(inputBatch)
        if args.bf16:
            # at::GradMode::is_enabled() will query a threadlocal flag
            # but new thread generate from throughputbench mark will 
            # init this flag to true, so we temporal cast embedding's
            # weight to bfloat16 for now
            if args.inference_only:
                dlrm.emb_l.bfloat16()
            dlrm = ipex.optimize(dlrm, dtype=torch.bfloat16, inplace=inplace)
        elif args.int8 and not args.tune:
            if IPEX_112:
                if args.num_cpu_cores != 0:
                    torch.set_num_threads(args.num_cpu_cores)
                qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
                    weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
                prepare(dlrm, qconfig, example_inputs=(X, lS_o, lS_i), inplace=True)
                dlrm.load_qconf_summary(qconf_summary = args.int8_configure)
                convert(dlrm, inplace=True)
                dlrm = torch.jit.trace(dlrm, [X, lS_o, lS_i])
                dlrm = torch.jit.freeze(dlrm)
            else:
                conf = ipex.quantization.QuantConf(args.int8_configure)
                dlrm = ipex.quantization.convert(dlrm, conf, (X, lS_o, lS_i))
        elif args.int8 and args.tune:
            dlrm = dlrm
        else:
            dlrm = ipex.optimize(dlrm, dtype=torch.float, inplace=inplace)
        if not IPEX_112:
            if args.int8 and not args.tune:
                dlrm = freeze(dlrm)
            else:
                with torch.cpu.amp.autocast(enabled=args.bf16):
                    dlrm = torch.jit.trace(dlrm, (X, lS_o, lS_i), check_trace=True)
                    dlrm = torch.jit.freeze(dlrm)
        dlrm(X, lS_o, lS_i)
        dlrm(X, lS_o, lS_i)
        return dlrm


def run_throughput_benchmark(args, dlrm, test_ld):
    if args.num_cpu_cores != 0:
        torch.set_num_threads(1)
    bench = ThroughputBenchmark(dlrm)
    for j, inputBatch in enumerate(test_ld):
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        bench.add_input(X, lS_o, lS_i)
        if j == 1000: 
            break
    stats = bench.benchmark(
        num_calling_threads=args.share_weight_instance,
        num_warmup_iters=100,
        num_iters=args.num_batches * args.share_weight_instance,
    )
    print(stats)
    latency = stats.latency_avg_ms
    throughput = (1 / latency) * 1000 * test_ld.dataset.batch_size * args.share_weight_instance
    print("throughput: {:.3f} fps".format(throughput))
    print("latency: {:.5f} ms".format(1/throughput * 1000))
    exit(0)


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    trace=True
):
    test_accu = 0
    test_samp = 0

    if args.print_auc:
        scores = []
        targets = []

    total_time = 0
    total_iter = 0
    if args.inference_only and trace:
        dlrm = trace_model(args, dlrm, test_ld)
    if args.share_weight_instance != 0:
        run_throughput_benchmark(args, dlrm, test_ld)
    with torch.cpu.amp.autocast(enabled=args.bf16):
        for i, testBatch in enumerate(test_ld):
            should_print = ((i + 1) % args.print_freq == 0 or i + 1 == len(test_ld)) and args.inference_only
            if should_print:
                gT = 1000.0 * total_time / total_iter
                print(
                    "Finished {} it {}/{}, {:.2f} ms/it,".format(
                        "inference", i + 1, len(test_ld), gT
                    ),
                    flush=True,
                )
                total_time = 0
                total_iter = 0
            # early exit if nbatches was set by the user and was exceeded
            if args.inference_only and nbatches > 0 and i >= nbatches:
                break

            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
                testBatch
            )

            # forward pass

            if not args.inference_only and isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                n_tables = lS_i_test.shape[0]
                idx = [lS_i_test[i] for i in range(n_tables)]
                offset = [lS_o_test[i] for i in range(n_tables)]
                include_last = [False for i in range(n_tables)]
                indices, offsets, indices_with_row_offsets = dlrm.emb_l.linearize_indices_and_offsets(idx, offset, include_last)

            start = time_wrap()
            if not args.inference_only and isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                Z_test = dlrm(X_test, indices, offsets, indices_with_row_offsets)
            else:
                Z_test = dlrm(X_test, lS_o_test, lS_i_test)

    
            total_time += (time_wrap() - start)
            total_iter += 1

            if args.print_auc:
                S_test = Z_test.detach().cpu().float().numpy()  # numpy array
                T_test = T_test.detach().cpu().float().numpy()  # numpy array
                scores.append(S_test)
                targets.append(T_test)
            elif not args.inference_only:
                with record_function("DLRM accuracy compute"):
                    # compute loss and accuracy
                    S_test = Z_test.detach().cpu().float().numpy()  # numpy array
                    T_test = T_test.detach().cpu().float().numpy()  # numpy array

                    mbs_test = T_test.shape[0]  # = mini_batch_size except last
                    A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                    test_accu += A_test
                    test_samp += mbs_test
            else:
                # do nothing to save time
                pass

    if args.print_auc:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
        acc_test = validation_results["accuracy"]
    elif not args.inference_only:
        acc_test = test_accu / test_samp
    else:
        pass

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
    }
    if not args.inference_only:
        model_metrics_dict["test_acc"] = acc_test

    if args.print_auc:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
        print("Accuracy: {:.34} ".format(validation_results["roc_auc"]))
    elif not args.inference_only:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        pass
    if not args.inference_only:
        return model_metrics_dict, is_best
    else:
        return validation_results["roc_auc"]


class DLRM_DataLoader(object):
    def __init__(self, loader=None):
        self.loader = loader
        self.batch_size = loader.dataset.batch_size
    def __iter__(self):
        for X_test, lS_o_test, lS_i_test, T in self.loader:
            yield (X_test, lS_o_test, lS_i_test), T


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # intel
    parser.add_argument("--print-auc", action="store_true", default=False)
    parser.add_argument("--should-test", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--share-weight-instance", type=int, default=0)
    parser.add_argument("--num-cpu-cores", type=int, default=0)
    parser.add_argument("--ipex-interaction", action="store_true", default=False)
    parser.add_argument("--ipex-merged-emb", action="store_true", default=False)
    parser.add_argument("--num-warmup-iters", type=int, default=1000)
    parser.add_argument("--int8", action="store_true", default=False)
    parser.add_argument("--int8-configure", type=str, default="./int8_configure.json")
    parser.add_argument("--dist-backend", type=str, default="ccl")
    parser.add_argument("--tune", action="store_true", default=False)

    global args
    global nbatches
    global nbatches_test
    args = parser.parse_args()
    ext_dist.init_distributed(backend=args.dist_backend)


    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size

    device = torch.device("cpu")
    print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
    nbatches_test = len(test_ld)

    ln_emb = train_data.counts
    # enforce maximum limit on number of vectors per embedding
    if args.max_ind_range > 0:
        ln_emb = np.array(
            list(
                map(
                    lambda x: x if x < args.max_ind_range else args.max_ind_range,
                    ln_emb,
                )
            )
        )
    else:
        ln_emb = np.array(ln_emb)
    m_den = train_data.m_den
    ln_bot[0] = m_den

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    # approach 1: all
    # num_int = num_fea * num_fea + m_den_out
    # approach 2: unique
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_threshold=args.loss_threshold,
    )
    if args.ipex_merged_emb:
        dlrm.emb_l = ipex.nn.modules.MergedEmbeddingBagWithSGD.from_embeddingbag_list(dlrm.emb_l, lr=args.learning_rate)
        dlrm.need_linearize_indices_and_offsets = torch.BoolTensor([False])

    if not args.inference_only:
        optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            args.lr_num_warmup_steps,
            args.lr_decay_start_step,
            args.lr_num_decay_steps,
        )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_train_loss = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_acc_test = ld_model["test_acc"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_acc_test = ld_acc_test
            total_loss = ld_total_loss
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}".format(
                ld_train_loss,
            )
        )
        print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))
        del ld_model

    ext_dist.barrier()
    print("time/loss/accuracy (if enabled):")

    if args.tune:
        from neural_compressor.experimental import Quantization, common

        def eval_func(model):
            args.int8 = False if model.ipex_config_path is None else True
            args.int8_configure = "" \
                if model.ipex_config_path is None else model.ipex_config_path
            with torch.no_grad():
                return inference(
                    args,
                    model,
                    best_acc_test,
                    best_auc_test,
                    test_ld,
                    trace=args.int8
                )

        assert args.inference_only, "Please set inference_only in arguments"
        quantizer = Quantization("./conf_ipex.yaml")
        quantizer.model = common.Model(dlrm)
        quantizer.calib_dataloader = DLRM_DataLoader(train_ld)
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        q_model.save(args.save_model)
        exit(0)

    if args.bf16 and not args.inference_only:
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
            if ext_dist.my_size > 1:
                local_bs = X.size()[0] // ext_dist.my_size
                rank_id = dlrm.rank
                X = X[rank_id * local_bs: (rank_id + 1) * local_bs]
                T = T[rank_id * local_bs: (rank_id + 1) * local_bs]
                global_bs = local_bs * ext_dist.my_size
                lS_o = lS_o[:, :global_bs]
                lS_i = lS_i[:, :global_bs]

            if isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                if ext_dist.my_size > 1:
                    batch_size = X.size()[0]
                    g_i = lS_i[dlrm.local_ln_emb]
                    g_o = lS_o[dlrm.local_ln_emb]
                    n_tables = g_i.shape[0]
                    idx = [g_i[i] for i in range(n_tables)]
                    offset = [g_o[i] for i in range(n_tables)]
                    include_last = [False for i in range(n_tables)]
                    indices, offsets, indices_with_row_offsets = dlrm.emb_l.linearize_indices_and_offsets(idx, offset, include_last)
                else:
                    n_tables = lS_i.shape[0]
                    idx = [lS_i[i] for i in range(n_tables)]
                    offset = [lS_o[i] for i in range(n_tables)]
                    include_last = [False for i in range(n_tables)]
                    indices, offsets, indices_with_row_offsets = dlrm.emb_l.linearize_indices_and_offsets(idx, offset, include_last)
            if isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                sample_input = (X, indices, offsets, indices_with_row_offsets)
            else:
                sample_input = (X, lS_o, lS_i)
            break
        dlrm, optimizer = ipex.optimize(dlrm, dtype=torch.bfloat16, optimizer=optimizer, inplace=True, sample_input=sample_input)

        if args.ipex_merged_emb:
            dlrm.emb_l.to_bfloat16_train()
        for i in range(len(dlrm.top_l)):
            if isinstance(dlrm.top_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                if isinstance(dlrm.top_l[i+1], torch.nn.ReLU):
                    dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'relu')
                else:
                    dlrm.top_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.top_l[i], 'sigmoid')
                dlrm.top_l[i + 1] = torch.nn.Identity()
        for i in range(len(dlrm.bot_l)):
            if isinstance(dlrm.bot_l[i], ipex.nn.utils._weight_prepack._IPEXLinear):
                if isinstance(dlrm.bot_l[i+1], torch.nn.ReLU):
                    dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'relu')
                else:
                    dlrm.bot_l[i] = ipex.nn.modules.IPEXLinearEltwise(dlrm.bot_l[i], 'sigmoid')
                dlrm.bot_l[i + 1] = torch.nn.Identity()

        if ext_dist.my_size > 1:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l)
    training_record = [0, 0]
    def update_training_performance(time, iters, training_record=training_record):
        if iters > args.num_warmup_iters:
            training_record[0] += time
            training_record[1] += 1

    def print_training_performance( training_record=training_record):
        if training_record[0] == 0:
            print("num-batches larger than warm up iters, please increase num-batches or decrease warmup iters")
            exit()
        total_samples = training_record[1] * args.mini_batch_size
        throughput = total_samples / training_record[0] * 1000
        print("throughput: {:.3f} fps".format(throughput))

    test_freq = args.test_freq if args.test_freq != -1  else nbatches // 20
    with torch.autograd.profiler.profile(
        enabled=args.enable_profiling, use_cuda=False, record_shapes=False
    ) as prof:
        if not args.inference_only:
            k = 0
            while k < args.nepochs:

                if k < skip_upto_epoch:
                    continue

                for j, inputBatch in enumerate(train_ld):

                    if j < skip_upto_batch:
                        continue

                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
                    if ext_dist.my_size > 1:
                        local_bs = X.size()[0] // ext_dist.my_size
                        rank_id = dlrm.rank
                        X = X[rank_id * local_bs: (rank_id + 1) * local_bs]
                        T = T[rank_id * local_bs: (rank_id + 1) * local_bs]
                        global_bs = local_bs * ext_dist.my_size
                        lS_o = lS_o[:, :global_bs]
                        lS_i = lS_i[:, :global_bs]

                    if isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                        if ext_dist.my_size > 1:
                            batch_size = X.size()[0]
                            g_i = lS_i[dlrm.local_ln_emb]
                            g_o = lS_o[dlrm.local_ln_emb]
                            n_tables = g_i.shape[0]
                            idx = [g_i[i] for i in range(n_tables)]
                            offset = [g_o[i] for i in range(n_tables)]
                            include_last = [False for i in range(n_tables)]
                            indices, offsets, indices_with_row_offsets = dlrm.emb_l.linearize_indices_and_offsets(idx, offset, include_last)
                        else:
                            n_tables = lS_i.shape[0]
                            idx = [lS_i[i] for i in range(n_tables)]
                            offset = [lS_o[i] for i in range(n_tables)]
                            include_last = [False for i in range(n_tables)]
                            indices, offsets, indices_with_row_offsets = dlrm.emb_l.linearize_indices_and_offsets(idx, offset, include_last)

                    t1 = time_wrap()

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

                    # forward pass
                    with torch.cpu.amp.autocast(enabled=args.bf16):
                        if isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                            Z = dlrm_wrap(
                                X,
                                indices,
                                offsets,
                                indices_with_row_offsets
                            ).float()
                        else:
                            Z = dlrm_wrap(
                                X,
                                lS_o,
                                lS_i,
                            ).float()

                    # loss
                    E = loss_fn_wrap(Z, T)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array

                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        optimizer.zero_grad(set_to_none=True)
                        # backward pass
                        E.backward()

                    with record_function("DLRM update"):
                        # optimizer
                        optimizer.step()
                    lr_scheduler.step()
                    if isinstance(dlrm.emb_l, ipex.nn.modules.MergedEmbeddingBagWithSGD):
                        dlrm.emb_l.sgd_args = dlrm.emb_l.sgd_args._replace(lr=lr_scheduler.get_last_lr()[0])

                    t2 = time_wrap()
                    total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    )
                    should_test = (
                        (args.should_test)
                        and (((j + 1) % test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )
                        update_training_performance(gT, j)

                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test:
                        model_metrics_dict, is_best = inference(
                            args,
                            dlrm,
                            best_acc_test,
                            best_auc_test,
                            test_ld,
                        )

                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                        if (
                            (args.mlperf_auc_threshold > 0)
                            and (best_auc_test > args.mlperf_auc_threshold)
                        ):
                            print(
                                "MLPerf testing auc threshold "
                                + str(args.mlperf_auc_threshold)
                                + " reached, stop training"
                            )
                k += 1  # nepochs
        else:
            print("Testing for inference only")
            with torch.no_grad():
                inference(
                    args,
                    dlrm,
                    best_acc_test,
                    best_auc_test,
                    test_ld
                )

    # profiling
    if not args.inference_only:
        print_training_performance()

    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
    exit(0)

if __name__ == "__main__":
    run()
