import numpy as np

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
    def __init__(self):
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
                 percentile=99.999):
        super(PercentileCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.percentile = percentile

    def collect_calib_data(self, datas):
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect_percentile(datas)
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
                 num_bins=128,
                 num_quantized_bins=128):
        super(EntropyCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
    
    def collect_calib_data(self, datas):
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect_entropy(datas)
        self.compute_entropy_range()

    def compute_entropy_range(self):
        # histogram_dict = self.histogram_dict
        histogram = self.collector.histogram
        self._calib_min, self._calib_max = self.get_entropy_threshold(histogram, self.num_quantized_bins)

    def get_entropy_threshold(self, histogram, num_quantized_bins):
        """Given a dataset, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref: 
        https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
        https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py
        """
        import copy
        from scipy.stats import entropy

        hist = histogram[0]
        hist_edges = histogram[1]
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2

        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(0, 0) for i in range(kl_divergence.size)]

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i
            end_index = zero_bin_index + i + 1 if (zero_bin_index + i + 1) <= num_bins else num_bins

            thresholds[i - num_half_quantized_bin] = (
                float(hist_edges[start_index]),
                float(hist_edges[end_index]),
            )

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy()  # a copy of np array
            left_outliers_count = sum(hist[:start_index])
            right_outliers_count = sum(hist[end_index:])
            p[0] += left_outliers_count
            p[-1] += right_outliers_count

            # nonzeros[i] incidates whether p[i] is non-zero
            nonzeros = (p != 0).astype(np.int64)

            # quantize p.size bins into quantized bins (default 128 bins)
            quantized_bins = np.zeros(num_quantized_bins, dtype=np.int64)
            num_merged_bins = sliced_distribution.size // num_quantized_bins

            # merge bins into quantized bins
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins
                quantized_bins[index] = sum(sliced_distribution[start:end])
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins :])

            # in order to compare p and q, we need to make length of q equals to length of p
            # expand quantized bins into p.size bins
            q = np.zeros(p.size, dtype=np.int64)
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins

                norm = sum(nonzeros[start:end])
                if norm != 0:
                    q[start:end] = float(quantized_bins[index]) / float(norm)

            p = smooth_distribution(p)
            q = smooth_distribution(q)

            if isinstance(q, np.ndarray):
                kl_divergence[i - num_half_quantized_bin] = entropy(p, q)
            else:
                kl_divergence[i - num_half_quantized_bin] = float("inf")

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx]
        min_value = histogram[2]
        max_value = histogram[3]
        if optimal_threshold[0] < min_value:
            optimal_threshold = (min_value, optimal_threshold[1])
        if optimal_threshold[1] > max_value:
            optimal_threshold = (optimal_threshold[0], max_value)
        return optimal_threshold[0], optimal_threshold[1]
        
class HistogramCollector():
    def __init__(self, num_bins=2048):
        self._num_bins = num_bins
        self._histogram = None
    
    def collect_percentile(self, data):
        data = np.asarray(data)
        data = data.flatten()
        assert data.size > 0, "collected intermediate data size"\
        "should not be 0, please check augmented_model"

        data = np.abs(data)
        max_range = np.max(data)
        min_range = np.min(data)
        
        if self._histogram is None:
            # first time it uses num_bins to compute histogram.
            width = max_range / self._num_bins
            bin_type = np.result_type(min_range, max_range, data)
            if np.issubdtype(bin_type, np.integer):
                bin_type = np.result_type(bin_type, float)
            calib_bin_edges = np.linspace(min_range, max_range, self._num_bins + 1, endpoint=True, dtype=bin_type)
            calib_hist, calib_bin_edges = np.histogram(data, bins=calib_bin_edges)
            self._histogram = (calib_hist, calib_bin_edges)
        else:
            calib_hist, calib_bin_edges = self._histogram
            width = calib_bin_edges[1] - calib_bin_edges[0]
            if max_range > calib_bin_edges[-1]:
                new_calib_bin_edges = np.arange(calib_bin_edges[-1] + width, max_range + width, width)
                calib_bin_edges = np.hstack((calib_bin_edges, new_calib_bin_edges))
            hist, calib_bin_edges = np.histogram(data, bins=calib_bin_edges)
            hist[:len(calib_hist)] += calib_hist
            calib_hist = hist
            self._histogram = (calib_hist, calib_bin_edges)
    
    def collect_entropy(self, data):
        data = np.asarray(data)
        data = data.flatten()
        assert data.size > 0, "collected intermediate data size"\
        "should not be 0, please check augmented_model"

        min_range = np.min(data)
        max_range = np.max(data)

        th = max(abs(min_range), abs(max_range))
        if self._histogram is None:
            hist, hist_edges = np.histogram(data, self._num_bins, range=(-th, th))
            self._histogram = (hist, hist_edges, min_range, max_range, th)
        else:
            self._histogram = self.combine_histogram(self._histogram, 
                                                    data, 
                                                    min_range, 
                                                    max_range, 
                                                    th)
                
    def combine_histogram(self, old_hist, data_arr, new_min, new_max, new_th):

        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist

        if new_th <= old_th:
            hist, _ = np.histogram(data_arr, 
                                   bins=len(old_hist), 
                                   range=(-old_th, old_th))
            return (
                old_hist + hist,
                old_hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                old_th,
            )
        else:
            # Need to generate new histogram with new_th
            if old_th == 0:
                hist, hist_edges = np.histogram(data_arr, len(old_hist), range=(-new_th, new_th))
                hist += old_hist
            else:
                old_num_bins = len(old_hist)
                old_step = 2 * old_th / old_num_bins
                half_increased_bins = int((new_th - old_th) // old_step + 1)
                new_num_bins = half_increased_bins * 2 + old_num_bins
                new_th = half_increased_bins * old_step + old_th
                hist, hist_edges = np.histogram(data_arr, 
                                                bins=new_num_bins, 
                                                range=(-new_th, new_th))
                hist[half_increased_bins:new_num_bins - half_increased_bins] += old_hist
            return (
                hist,
                hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                new_th,
            )

    def reset(self):
        """Reset the collected histogram"""
        self._histogram = {}

    @property
    def histogram(self):
        return self._histogram


def smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref: 
    http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py
    """

    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
        # raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        return -1
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, "n_zeros=%d, n_nonzeros=%d, eps1=%f" % (
        n_zeros,
        n_nonzeros,
        eps1,
    )

    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0

    return hist