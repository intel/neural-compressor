import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))

from parts.manifest import Manifest
from parts.segment import AudioSegment

import numpy as np

import mlperf_loadgen as lg


class AudioQSL:
    def __init__(self, dataset_dir, manifest_filepath, labels,
                 sample_rate=16000, perf_count=None):
        m_paths = [manifest_filepath]
        self.manifest = Manifest(dataset_dir, m_paths, labels, len(labels),
                                 normalize=True, max_duration=15.0)
        self.sample_rate = sample_rate
        self.count = len(self.manifest)
        perf_count = self.count if perf_count is None else perf_count
        self.sample_id_to_sample = {}
        self.qsl = lg.ConstructQSL(self.count, perf_count,
                                   self.load_query_samples,
                                   self.unload_query_samples)
        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours. Number of samples: {2}".format(
                self.manifest.duration / 3600,
                self.manifest.filtered_duration / 3600,
                self.count))

    def load_query_samples(self, sample_list):
        for sample_id in sample_list:
            self.sample_id_to_sample[sample_id] = self._load_sample(sample_id)

    def unload_query_samples(self, sample_list):
        for sample_id in sample_list:
            del self.sample_id_to_sample[sample_id]

    def _load_sample(self, index):
        sample = self.manifest[index]
        segment = AudioSegment.from_file(sample['audio_filepath'][0],
                                         target_sr=self.sample_rate)
        waveform = segment.samples
        assert isinstance(waveform, np.ndarray) and waveform.dtype == np.float32
        return waveform

    def __getitem__(self, index):
        return self.sample_id_to_sample[index]

    def __del__(self):
        lg.DestroyQSL(self.qsl)
        print("Finished destroying QSL.")

# We have no problem fitting all data in memory, so we do that, in
# order to speed up execution of the benchmark.
class AudioQSLInMemory(AudioQSL):
    def __init__(self, dataset_dir, manifest_filepath, labels,
                 sample_rate=16000, perf_count=None):
        super().__init__(dataset_dir, manifest_filepath, labels,
                         sample_rate, perf_count)
        super().load_query_samples(range(self.count))

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass
