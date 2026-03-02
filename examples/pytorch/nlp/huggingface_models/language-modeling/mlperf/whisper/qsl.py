# Copyright 2025 The MLPerf Authors. All Rights Reserved.
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

# Standard packages
import sys
import os
from multiprocessing import Pool

# Installed packages
import numpy as np
import librosa

# Local python packages
import mlperf_loadgen as lg
from manifest import Manifest


Manifest_Global = None
max_duration = float(os.environ.get("MAX_DURATION", "30.0"))

def load_sample_from_file(index):
    global Manifest
    sample = Manifest_Global[index]
    filepath = sample['audio_filepath'][0]
    prompt = {
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": librosa.load(filepath, sr=16000),
        },
    }
    duration = sample['duration']
    return prompt

class AudioQSL:
    def __init__(self, dataset_dir, manifest_filepath, labels,
                 sample_rate=16000, perf_count=None, skip_qsl=False):
        global Manifest_Global
        m_paths = [manifest_filepath]
        self.manifest = Manifest(dataset_dir, m_paths, labels, len(labels), max_duration=max_duration)
        Manifest_Global = self.manifest
        self.sample_rate = sample_rate
        self.count = len(self.manifest)
        perf_count = self.count if perf_count is None else perf_count
        self.sample_id_to_sample = {}
        self.loaded = False
        if skip_qsl:
            self.qsl = None
        else:
            self.qsl = lg.ConstructQSL(self.count, perf_count,
                                    self.load_query_samples,
                                    self.unload_query_samples)

        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours. Number of samples: {2}".format(
                self.manifest.duration / 3600,
                self.manifest.filtered_duration / 3600,
                self.count))
    
    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass

    def __getitem__(self, index):
        return self.sample_id_to_sample[index]

    def __del__(self):
        lg.DestroyQSL(self.qsl)
        print("Finished destroying QSL.")

# We have no problem fitting all data in memory, so we do that, in
# order to speed up execution of the benchmark.
class AudioQSLInMemory(AudioQSL):
    def __init__(self, dataset_dir, manifest_filepath, labels,
                 sample_rate=16000, perf_count=None, skip_qsl=True):
        super().__init__(dataset_dir, manifest_filepath, labels,
                         sample_rate, perf_count)
        self.load_query_samples(range(self.count))

    def load_query_samples(self, sample_list):
        if not self.loaded:
            pool = Pool(8)
            print("pool size 8")
            result = pool.map(load_sample_from_file, sample_list)
            for sample_id in sample_list:
                self.sample_id_to_sample[sample_id] = result[sample_id]
            pool.close()
            pool.join()
            self.loaded = True

    def unload_query_samples(self, sample_list):
        for sample_id in sample_list:
            del self.sample_id_to_sample[sample_id]
    def __del__(self):
        print("FInished destroying no QSL")
