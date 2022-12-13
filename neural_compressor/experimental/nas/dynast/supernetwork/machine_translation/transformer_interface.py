# https://github.com/mit-han-lab/hardware-aware-transformers/blob/master/LICENSE
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translate pre-processed data with a trained model."""
import time
import warnings

import numpy as np

from neural_compressor.utils.utility import logger, LazyImport

from .transformer_supernetwork import TransformerSuperNetwork

torch = LazyImport('torch')
torchprofile = LazyImport('torchprofile')
fairseq = LazyImport('fairseq')

warnings.filterwarnings("ignore")


def compute_bleu(config, dataset_path, checkpoint_path):
    """Measure BLEU score of the Transformer-based model."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks
    MosesTokenizer = fairseq.data.encoders.moses_tokenizer.MosesTokenizer
    StopwatchMeter = fairseq.meters.StopwatchMeter
    progress_bar = fairseq.progress_bar

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser, [dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = 128
    args.eval_bleu_remove_bpe = '@@ '
    args.eval_bleu_detok = 'moses'

    utils.import_user_module(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # when running on CPU, use fp32 as default
    if not use_cuda:
        args.fp16 = False

    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    tokenizer = MosesTokenizer(args)
    task.tokenizer=tokenizer
    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    model = TransformerSuperNetwork(task)
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(state['model'],
                          strict=True)

    if use_cuda:
        model.cuda()
    model.set_sample_config(config)
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()


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
    generator = task.build_generator([model], args)

    num_sentences = 0
    bleu_list = []
    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            bleu = task._inference_with_bleu(generator,sample,model)
            bleu_list.append(bleu.score)

            num_sentences += sample['nsentences']

    bleu_score = np.mean(np.array(bleu_list))
    return bleu_score


def compute_latency(config, dataset_path, batch_size, get_model_parameters=False):
    """Measure latency of the Transformer-based model."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks

    parser = options.get_generation_parser()

    args = options.parse_args_and_arch(parser, [dataset_path])

    args.data = dataset_path
    args.beam = 5
    args.remove_bpe = '@@ '
    args.gen_subset = 'test'
    args.lenpen = 0.6
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.batch_size = batch_size
    utils.import_user_module(args)
    args.latgpu = False
    args.latcpu = True
    args.latiter = 100

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Load ensemble
    model = TransformerSuperNetwork(task)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}

    dummy_sentence_length = dummy_sentence_length_dict['wmt']

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor(
        [dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([dummy_sentence_length])
    prev_output_tokens_test_with_beam = torch.tensor(
        [dummy_prev] * args.beam, dtype=torch.long)
    bsz = 1
    new_order = torch.arange(bsz).view(-1, 1).repeat(1,
                                                     args.beam).view(-1).long()
    if args.latcpu:
        model.cpu()
        logger.info('Measuring model latency on CPU for dataset generation...')
    elif args.latgpu:
        model.cuda()
        src_tokens_test = src_tokens_test
        src_lengths_test = src_lengths_test
        prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam
        logger.info('Measuring model latency on GPU for dataset generation...')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    model.set_sample_config(config)

    model.eval()

    with torch.no_grad():

        # dry runs
        for _ in range(15):
            encoder_out_test = model.encoder(
                src_tokens=src_tokens_test, src_lengths=src_lengths_test)

        encoder_latencies = []
        logger.info('[DyNAS-T] Measuring encoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                start = time.time()
            elif args.latcpu:
                start = time.time()

            model.encoder(src_tokens=src_tokens_test,
                          src_lengths=src_lengths_test)

            if args.latgpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)
            elif args.latcpu:
                end = time.time()
                encoder_latencies.append((end - start) * 1000)

        encoder_latencies.sort()
        encoder_latencies = encoder_latencies[int(
            args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
        logger.info(
            f'[DyNAS-T] Encoder latency for dataset generation: Mean: '
            '{np.mean(encoder_latencies)} ms; Std: {np.std(encoder_latencies)} ms'
        )

        encoder_out_test_with_beam = model.encoder.reorder_encoder_out(
            encoder_out_test, new_order)

        # dry runs
        for _ in range(15):
            model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                          encoder_out=encoder_out_test_with_beam)

        # decoder is more complicated because we need to deal with incremental states and auto regressive things
        decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}

        decoder_iterations = decoder_iterations_dict['wmt']
        decoder_latencies = []

        logger.info('[DyNAS-T] Measuring decoder for dataset generation...')
        for _ in range(args.latiter):
            if args.latgpu:
                start = time.time()
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
        decoder_latencies = decoder_latencies[int(
            args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

    logger.info(
        f'[DyNAS-T] Decoder latency for dataset generation: Mean: '
        '{np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms'
    )

    lat_mean = np.mean(encoder_latencies)+np.mean(decoder_latencies)
    lat_std = np.std(encoder_latencies)+np.std(decoder_latencies)
    return lat_mean, lat_std


def compute_macs(config, dataset_path):
    """Calculate MACs for Transformer-based models."""
    options = fairseq.options
    utils = fairseq.utils
    tasks = fairseq.tasks

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

    # Load model
    logger.info('[DyNAS-T] loading model(s) from {}'.format(args.path))
    model = TransformerSuperNetwork(task)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}

    dummy_sentence_length = dummy_sentence_length_dict['wmt']


    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    model.eval()
    model.profile(mode=True)
    model.set_sample_config(config)
    macs = torchprofile.profile_macs(model, args=(torch.tensor([dummy_src_tokens], dtype=torch.long),
                                   torch.tensor([30]), torch.tensor([dummy_prev], dtype=torch.long)))

    model.profile(mode=False)

    return macs
