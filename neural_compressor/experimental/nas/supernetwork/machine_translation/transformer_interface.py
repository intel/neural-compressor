"""
Translate pre-processed data with a trained model.
"""
import torch

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import sys
import pdb
import numpy as np
import subprocess
import os
from fairseq.data import dictionary
import csv
import json
import warnings
from .transformer_supernetwork import TransformerSuperNetwork

import sys
import logging
import tqdm
import time
import copy
from datetime import datetime
import ctypes
import math
warnings.filterwarnings("ignore")


try:
    from fairseq import libbleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: missing libbleu.so. run `pip install --editable .`\n')
    raise e


C = ctypes.cdll.LoadLibrary(libbleu.__file__)


class BleuStat(ctypes.Structure):
    _fields_ = [
        ('reflen', ctypes.c_size_t),
        ('predlen', ctypes.c_size_t),
        ('match1', ctypes.c_size_t),
        ('count1', ctypes.c_size_t),
        ('match2', ctypes.c_size_t),
        ('count2', ctypes.c_size_t),
        ('match3', ctypes.c_size_t),
        ('count3', ctypes.c_size_t),
        ('match4', ctypes.c_size_t),
        ('count4', ctypes.c_size_t),
    ]


class Scorer(object):
    def __init__(self, pad, eos, unk):
        self.stat = BleuStat()
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.reset()

    def reset(self, one_init=False):
        if one_init:
            C.bleu_one_init(ctypes.byref(self.stat))
        else:
            C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError('ref must be a torch.IntTensor (got {})'
                            .format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError('pred must be a torch.IntTensor(got {})'
                            .format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos))

    def score(self, order=4):
        psum = sum(math.log(p) if p > 0 else float('-Inf')
                   for p in self.precision()[:order])
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]

    def brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def result_string(self, order=4):
        assert order <= 4, "BLEU scores for order > 4 aren't supported"
        fmt = 'BLEU{} = {:2.2f}, {:2.1f}'
        for _ in range(1, order):
            fmt += '/{:2.1f}'
        fmt += ' (BP={:.3f}, ratio={:.3f}, syslen={}, reflen={})'
        bleup = [p * 100 for p in self.precision()[:order]]
        return fmt.format(order, self.score(order=order), *bleup,
                          self.brevity(), self.stat.predlen/self.stat.reflen,
                          self.stat.predlen, self.stat.reflen)


def get_bleu_score(args,ref,sys):
    dict = dictionary.Dictionary()
    order =4
    sacrebleu = False
    sentence_bleu = False
    ignore_case = False
    def readlines(fd):
        for line in fd.readlines():
            if ignore_case:
                yield line.lower()
            else:
                yield line


    if sentence_bleu:
        def score(fdsys):
            with open(ref) as fdref:
                scorer = Scorer(dict.pad(), dict.eos(), dict.unk())
                for i, (sys_tok, ref_tok) in enumerate(zip(readlines(fdsys), readlines(fdref))):
                    scorer.reset(one_init=True)
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                    print(i, scorer.result_string(order))
    else:
        def score(fdsys):
            with open(ref) as fdref:
                scorer = Scorer(dict.pad(), dict.eos(), dict.unk())
                for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                    sys_tok = dict.encode_line(sys_tok)
                    ref_tok = dict.encode_line(ref_tok)
                    scorer.add(ref_tok, sys_tok)
                print(scorer.result_string(order))
                return(scorer.score(order))


    if sys == '-':
        score = score(sys.stdin)
    else:
        with open(sys, 'r') as f:
            score = score(f)
    return score

def compute_bleu(config,dataset_path,checkpoint_path):

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser,[dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = 128
    utils.import_user_module(args)
    max_tokens = 12000


    use_cuda = torch.cuda.is_available() and not args.cpu

    # when running on CPU, use fp32 as default
    if not use_cuda:
        args.fp16 = False

    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary


    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model = TransformerSuperNetwork(task)
    state = torch.load(checkpoint_path,map_location=torch.device('cpu'))

    model.load_state_dict(state['model'],
                            strict=True)

    if use_cuda:
        model.cuda()
    print(config)
    model.set_sample_config(config)
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    print(args.path, file=sys.stderr)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=128,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions()]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator([model],args)

    num_sentences = 0
    has_target = True
    decoder_times_all = []
    input_len_all = []
    with open('translations_out.txt','a') as fname_translations:
        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = TimeMeter()
            for sample in t:

                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = sample['target'][:, :args.prefix_size]

                gen_timer.start()
                hypos = task.inference_step(generator, [model], sample, prefix_tokens)
                input_len_all.append(np.mean(sample['net_input']['src_lengths'].cpu().numpy()))
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                gen_timer.stop(num_generated_tokens)

                for i, sample_id in enumerate(sample['id'].tolist()):
                    has_target = sample['target'] is not None

                    # Remove padding
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        if has_target:
                            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                    if not args.quiet:
                        if src_dict is not None:
                            #print('S-{}\t{}'.format(sample_id, src_str))
                            fname_translations.write('S-{}\t{}'.format(sample_id, src_str))
                            fname_translations.write('\n')

                        if has_target:
                            #print('T-{}\t{}'.format(sample_id, target_str))
                            fname_translations.write('T-{}\t{}'.format(sample_id, target_str))
                            fname_translations.write('\n')

                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )

                        if not args.quiet:
                           
                            fname_translations.write('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                            fname_translations.write('\n')
                            fname_translations.write('P-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(map(
                                        lambda x: '{:.4f}'.format(x),
                                        hypo['positional_scores'].tolist(),
                                    ))
                                ))
                            fname_translations.write('\n')

                            if args.print_alignment:
                                fname_translations.write('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                                ))
                                fname_translations.write('\n')

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += sample['nsentences']


    os.system("grep ^H translations_out.txt | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > sys.txt")
    os.system("grep ^T translations_out.txt | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ref.txt")
    bleu_score = get_bleu_score(args,"ref.txt","sys.txt")
    print(bleu_score)

    os.system("rm ref.txt")
    os.system("rm sys.txt")
    os.system("rm translations_out.txt")
    return bleu_score

def compute_latency(config,dataset_path,get_model_parameters=False):
    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser,[dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = 128
    utils.import_user_module(args)
    max_tokens = 12000
    args.latgpu=False
    args.latcpu=True
    args.latiter=100

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    #Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary


    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model = TransformerSuperNetwork(task)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}

    dummy_sentence_length = dummy_sentence_length_dict['wmt']


    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)#.cuda()
    src_lengths_test = torch.tensor([dummy_sentence_length])#.cuda()
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)#.cuda()
    bsz = 1
    new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()#.cuda()
    if args.latcpu:
        model.cpu()
        print('Measuring model latency on CPU for dataset generation...')
    elif args.latgpu:
        model.cuda()
        src_tokens_test = src_tokens_test#.cuda()
        src_lengths_test = src_lengths_test#.cuda()
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam#.cuda()
        print('Measuring model latency on GPU for dataset generation...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)


    model.set_sample_config(config)
    
    model.eval()
   
    with torch.no_grad():

        # dry runs
        for _ in range(15):
            encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        encoder_latencies = []
        print('Measuring encoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                #start.record()
                start = time.time()
            elif args.latcpu:
                start = time.time()

            model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            if args.latgpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)
            elif args.latcpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)

        encoder_latencies.sort()
        encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
        print(f'Encoder latency for dataset generation: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')


        encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

        # dry runs
        for _ in range(15):
            model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                               encoder_out=encoder_out_test_with_beam)

        # decoder is more complicated because we need to deal with incremental states and auto regressive things
        decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}

        decoder_iterations = decoder_iterations_dict['wmt']
        print(decoder_iterations)
        decoder_latencies = []
        print('Measuring decoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                start = time.time()
                #start.record()
            elif args.latcpu:
                start = time.time()
            incre_states = {}
            for k_regressive in range(decoder_iterations):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                                   encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
            if args.latgpu:
                end = time.time()
                decoder_latencies.append((end - start) * 1000)

            elif args.latcpu:
                end = time.time()
                decoder_latencies.append((end - start) * 1000)

        # only use the 10% to 90% latencies to avoid outliers
        decoder_latencies.sort()
        decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

    print(decoder_latencies)
    print(f'Decoder latency for dataset generation: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms')

    lat_mean = np.mean(encoder_latencies)+np.mean(decoder_latencies)
    lat_std = np.std(encoder_latencies)+np.std(decoder_latencies)
    return lat_mean, lat_std
