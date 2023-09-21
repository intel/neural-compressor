# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

import array
import torch
import numpy as np
import toml
import mlperf_loadgen as lg
from tqdm import tqdm

from QSL import AudioQSL, AudioQSLInMemory
from decoders import ScriptGreedyDecoder
from helpers import add_blank_label
from preprocessing import AudioPreprocessing
from model_separable_rnnt import RNNT


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


class PytorchSUT:
    def __init__(self, config_toml, checkpoint_path, dataset_dir,
                 manifest_filepath, perf_count):
        config = toml.load(config_toml)

        dataset_vocab = config['labels']['labels']
        rnnt_vocab = add_blank_label(dataset_vocab)
        featurizer_config = config['input_eval']

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = AudioQSLInMemory(dataset_dir,
                                    manifest_filepath,
                                    dataset_vocab,
                                    featurizer_config["sample_rate"],
                                    perf_count)
        self.audio_preprocessor = AudioPreprocessing(**featurizer_config)
        self.audio_preprocessor.eval()
        self.audio_preprocessor = torch.jit.script(self.audio_preprocessor)
        self.audio_preprocessor = torch.jit._recursive.wrap_cpp_module(
            torch._C._freeze_module(self.audio_preprocessor._c))

        model = RNNT(
            feature_config=featurizer_config,
            rnnt=config['rnnt'],
            num_classes=len(rnnt_vocab)
        )
        model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path),
                              strict=True)
        model.eval()
        self.greedy_decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            waveform = self.qsl[query_sample.index]
            assert waveform.ndim == 1
            waveform_length = np.array(waveform.shape[0], dtype=np.int64)
            waveform = np.expand_dims(waveform, 0)
            waveform_length = np.expand_dims(waveform_length, 0)
            with torch.no_grad():
                waveform = torch.from_numpy(waveform)
                waveform_length = torch.from_numpy(waveform_length)
                feature, feature_length = self.audio_preprocessor.forward((waveform, waveform_length))
                assert feature.ndim == 3
                assert feature_length.ndim == 1
                feature = feature.permute(2, 0, 1)

                _, _, transcript = self.greedy_decoder.forward(feature, feature_length)

            assert len(transcript) == 1
            response_array = array.array('q', transcript[0])
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, bi[0],
                                              bi[1] * response_array.itemsize)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        lg.DestroySUT(self.sut)
        print("Finished destroying SUT.")