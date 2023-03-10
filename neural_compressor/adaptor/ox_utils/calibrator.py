import numpy as np
from collections import Counter
from scipy.stats import entropy

CALIBRATOR = {}

def calib_registry(calib_method):
    def decorator_calib(cls):
        assert cls.__name__.endswith(
            'Calibrator'), "The name of subclass of Calibrator should end with \'Calibrator\' substring."
        if cls.__name__[:-len('Calibrator')] in CALIBRATOR: # pragma: no cover
            raise ValueError('Cannot have two operators with the same name.')
        CALIBRATOR[calib_method.strip()] = cls
        return cls
    return decorator_calib

class CalibratorBase:
    def __init__(self):
        self._calib_min = None
        self._calib_max = None

    def collect(self, datas):
        self.collect_calib_data(datas)

    def clear(self):
        self._calib_min = None
        self._calib_max = None
    
    @property
    def calib_range(self):
        return self._calib_min, self._calib_max

@calib_registry(calib_method='minmax')
class MinMaxCalibrator(CalibratorBase):
    def __init__(self, **kwargs):
        super(MinMaxCalibrator, self).__init__()
    
    def collect_calib_data(self, datas):
        for data in datas:
            local_min = np.min(data)
            local_max = np.max(data)
            if self._calib_min is None and self._calib_max is None:
                self._calib_min = local_min
                self._calib_max = local_max
            else:
                self._calib_min = np.minimum(self._calib_min, local_min)
                self._calib_max = np.maximum(self._calib_max, local_max)

@calib_registry(calib_method='percentile')
class PercentileCalibrator(CalibratorBase):
    def __init__(self, 
                 num_bins=2048,
                 percentile=99.999,
                 **kwargs):
        super(PercentileCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.percentile = percentile

    def collect_calib_data(self, datas):
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect(datas)
        self.compute_percentile_range(self.percentile)

    def compute_percentile_range(self, percentile):
        if percentile < 0 or percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        calib_hist, calib_bin_edges = self.collector.histogram
        total = calib_hist.sum()
        cdf = np.cumsum(calib_hist / total)
        idx = np.searchsorted(cdf, percentile / 100)
        self._calib_min = -calib_bin_edges[idx].astype('float32')
        self._calib_max = calib_bin_edges[idx].astype('float32')

@calib_registry(calib_method='kl')
class EntropyCalibrator(CalibratorBase):
    def __init__(self, 
                 num_bins=2048,
                 start_bin=128,
                 unsigned=False,
                 **kwargs):
        super(EntropyCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.start_bin = start_bin
        self.unsigned =unsigned
        self.stride = kwargs.pop('stride', 1)
        self.num_bits = kwargs.pop('num_bits', 8)
        
    def collect_calib_data(self, datas):
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect(datas)
        self.compute_entropy_range()

    def compute_entropy_range(self):
        calib_hist, calib_bin_edges = self.collector.histogram
        self._calib_min, self._calib_max = self.compute_hist(calib_hist, calib_bin_edges)
            
    def compute_hist(self, calib_hist, calib_bin_edges):
        bins = calib_hist[:]
        bins[0] = bins[1]
        total_data = np.sum(bins)
        divergences = []
        arguments = []

        nbins = 1 << (self.num_bits - 1 + int(self.unsigned))
        stop = len(bins)

        new_density_counts = np.zeros(nbins, dtype=np.float64)
        for i in range(self.start_bin, stop + 1, self.stride):
            new_density_counts.fill(0)
            space = np.linspace(0, i, num=nbins + 1)
            digitized_space = np.digitize(range(i), space) - 1

            digitized_space[bins[:i] == 0] = -1

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bins[idx]

            counter = Counter(digitized_space)
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            new_density = np.zeros(i, dtype=np.float64)
            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = np.sum(new_density) + np.sum(bins[i:])
            new_density = new_density / np.sum(new_density)

            reference_density = np.array(bins[:len(digitized_space)])
            reference_density[-1] += np.sum(bins[i:])

            total_counts_old = np.sum(reference_density)
            if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
                raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
                    total_counts_new, total_counts_old, total_data))
            reference_density = reference_density / np.sum(reference_density)

            ent = entropy(reference_density, new_density)
            divergences.append(ent)
            arguments.append(i)

        divergences = np.array(divergences)
        last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
        calib_max = calib_bin_edges[last_argmin * self.stride + self.start_bin]
        return -calib_max.astype('float32'), calib_max.astype('float32')


class HistogramCollector():
    def __init__(self, num_bins=2048):
        self._num_bins = num_bins
        self._histogram = None
    
    def collect(self, datas):
        for data in datas:
            data = np.abs(data)
            temp_max = np.max(data)
            temp_mix = np.min(data)
            if self._histogram is None:
                # first time it uses num_bins to compute histogram.
                width = temp_max / self._num_bins
                bin_type = np.result_type(temp_mix, temp_max, data)
                if np.issubdtype(bin_type, np.integer):
                    bin_type = np.result_type(bin_type, float)
                calib_bin_edges = np.linspace(temp_mix, temp_max, self._num_bins + 1, endpoint=True, dtype=bin_type)
                calib_hist, calib_bin_edges = np.histogram(data, bins=calib_bin_edges)
                self._histogram = (calib_hist, calib_bin_edges)
            else:
                calib_hist, calib_bin_edges = self._histogram
                width = calib_bin_edges[1] - calib_bin_edges[0]
                if temp_max > calib_bin_edges[-1]:
                    new_calib_bin_edges = np.arange(calib_bin_edges[-1] + width, temp_max + width, width)
                    calib_bin_edges = np.hstack((calib_bin_edges, new_calib_bin_edges))
                hist, calib_bin_edges = np.histogram(data, bins=calib_bin_edges)
                hist[:len(calib_hist)] += calib_hist
                calib_hist = hist
                self._histogram = (calib_hist, calib_bin_edges)

    def reset(self):
        """Reset the collected histogram"""
        self._histogram_dict = {}

    @property
    def histogram(self):
        return self._histogram
