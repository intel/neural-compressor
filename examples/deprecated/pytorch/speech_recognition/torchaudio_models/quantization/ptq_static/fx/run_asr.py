# Copyright 2020 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import torch
import torchaudio
from jiwer import wer
import argparse
import logging
import time
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--root', metavar='DIR', required=True, help='path to dataset')
parser.add_argument('--url', choices=['dev-clean', 'dev-other', 'test-clean', 'test-other', 
                                      'train-clean-100', 'train-clean-360', 'train-other-500'],
                    default='test-clean',
                    help='the dataset form')
parser.add_argument('--folder_in_archive', default='LibriSpeech')
parser.add_argument('--model', choices=['wav2vec2', 'hubert'], default='wav2vec2')
parser.add_argument('--batch_size', type=int,default=1)
parser.add_argument('--int8', dest='int8', action='store_true')
parser.add_argument('--tune', dest='tune', action='store_true', 
                    help='tune best int8 model with Neural Compressor on calibration dataset')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument('--accuracy_only', dest='accuracy_only', action='store_true',
                    help='For accuracy measurement only.')
parser.add_argument('--tuned_checkpoint', default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Neural Compressor (default: ./)')
parser.add_argument('--iters', default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('--warmup_iter', default=5, type=int,
                    help='For benchmark measurement only.')
model_dict = {
                'wav2vec2': torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H,
                'hubert': torchaudio.pipelines.HUBERT_ASR_LARGE
                }

#decoder
class GreedyCTCDecoder(torch.nn.Module):
  def __init__(self, labels, blank=0):
    super().__init__()
    self.labels = labels
    self.blank = blank

  def forward(self, emission: torch.Tensor) -> str:
    """Given a sequence emission over labels, get the best path string
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

    Returns:
      str: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [i for i in indices if i != self.blank]
    return ''.join([self.labels[i] for i in indices])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = model_dict[args.model]
    model = bundle.get_model().to(device)
    logger.info("Sample Rate:", bundle.sample_rate)
    logger.info("Labels:", bundle.get_labels())
    logger.info(model.__class__)

    #prepare dataset
    dataset = torchaudio.datasets.LIBRISPEECH(args.root, args.url, args.folder_in_archive)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())

    def eval_func(model):
        predict = []
        text = []
        with torch.inference_mode():
            for index, wave in enumerate(val_dataloader):
                emission, _ = model(wave[0][0])
                transcript = decoder(emission[0])
                predict.append(transcript)
                text.append(wave[2][0])
                prediction = [pre.replace("|", " ")for pre in predict]
                WER = wer(text, prediction)
            print("Accuracy: %.5f" % (1-WER))
        return 1-WER
    
    #tune
    if args.tune:
        def calib_func(model):
            for index, wave in enumerate(val_dataloader):
                model(wave[0][0])
                if index == 100:
                    break

        from neural_compressor import PostTrainingQuantConfig, quantization
        for index, wave in enumerate(val_dataloader):
            example_inputs = wave[0][0]
            break
        conf = PostTrainingQuantConfig(approach="static",
                                       example_inputs=example_inputs)
        q_model = quantization.fit(model,
                                   conf=conf,
                                   eval_func=eval_func,
                                   calib_func=calib_func,
                                   calib_dataloader=val_dataloader
                                   )
        q_model.save(args.tuned_checkpoint)
        exit(0)
    
    #benchmark
    if args.int8:
        from neural_compressor.utils.pytorch import load
        model = load(
                os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model)

    if args.benchmark:
        def b_func(model):
            predict = []
            text = []
            results = {}
            batch_time = AverageMeter('Time', ':6.3f')
            with torch.inference_mode():
                for i, wave in enumerate(val_dataloader): 
                    if i >= args.warmup_iter:
                        start = time.time()
                    emission, _ = model(wave[0][0])
                    transcript = decoder(emission[0])
                    # measure elapsed time
                    if i >= args.warmup_iter:
                        batch_time.update(time.time() - start)
                    predict.append(transcript)
                    text.append(wave[2][0])
                    if args.iters > 0 and i >= (args.warmup_iter + args.iters - 1):
                        break
                prediction = [pre.replace("|", " ")for pre in predict]
                WER = wer(text, prediction)
            results['WER'] = WER
            results['accuracy'] = 1 - WER
            results['average_batch_time'] = batch_time.avg
            print("Accuracy: %.5f" % results['accuracy'])
            print('Latency: %.3f ms' % (results['average_batch_time'] * 1000))
            print('Throughput: %.3f images/sec' % (args.batch_size / results['average_batch_time']))
            print('Batch size = %d' % args.batch_size)
            return results['accuracy']

        from neural_compressor.config import BenchmarkConfig
        from neural_compressor import benchmark
        b_conf = BenchmarkConfig(
                                 cores_per_instance=4,
                                 num_of_instance=1
                                 )
        benchmark.fit(model, b_conf, b_func=b_func)
        exit(0)

    if args.accuracy_only:
        eval_func(model)
        exit(0)


if __name__ == "__main__":
    main()
 
