#!/usr/bin/env python
# coding: utf-8
#
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
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Calibrator for onnx models."""

import numpy as np

CALIBRATOR = {}


def calib_registry(calib_method):
    """The class decorator used to register all Calibrator subclasses."""

    def decorator_calib(cls):
        assert cls.__name__.endswith(
            "Calibrator"
        ), "The name of subclass of Calibrator should end with 'Calibrator' substring."
        if cls.__name__[: -len("Calibrator")] in CALIBRATOR:  # pragma: no cover
            raise ValueError("Cannot have two operators with the same name.")
        CALIBRATOR[calib_method.strip()] = cls
        return cls

    return decorator_calib


class CalibratorBase:
    """Base calibrator class."""

    def __init__(self):
        """Initialize base calibrator class."""
        self._calib_min = None
        self._calib_max = None

    def collect(self, datas):
        """Collect calibration range."""
        self.collect_calib_data(datas)

    def clear(self):
        """Clear calibration range."""
        self._calib_min = None
        self._calib_max = None

    def collect_calib_data(self, datas):
        """Collect calibration range value."""
        raise NotImplementedError

    @property
    def calib_range(self):
        """Get calibration range value."""
        return self._calib_min, self._calib_max


@calib_registry(calib_method="minmax")
class MinMaxCalibrator(CalibratorBase):
    """MinMax calibrator class."""

    def __init__(self):
        """Initialize minmax calibrator class."""
        super(MinMaxCalibrator, self).__init__()

    def collect_calib_data(self, datas):
        """Collect calibration range."""
        if isinstance(datas, list) and len(set([data.shape for data in datas])) != 1:
            for data in datas:
                if data.size == 0:  # pragma: no cover
                    continue
                self._collect_value(data)
        else:
            datas = np.asarray(datas)
            datas = datas.flatten()
            assert datas.size > 0, "collected intermediate data size" "should not be 0, please check augmented_model"
            self._collect_value(datas)

    def _collect_value(self, data):
        """Collect min/max value."""
        data = np.asarray(data)

        local_min = np.min(data[np.isinf(data) == False])  # noqa: E712
        local_max = np.max(data[np.isinf(data) == False])  # noqa: E712
        if self._calib_min is None and self._calib_max is None:
            self._calib_min = local_min
            self._calib_max = local_max
        else:
            self._calib_min = np.minimum(self._calib_min, local_min)
            self._calib_max = np.maximum(self._calib_max, local_max)

    @property
    def method_name(self):
        """Get calibration method name."""
        return "minmax"


@calib_registry(calib_method="percentile")
class PercentileCalibrator(CalibratorBase):
    """Percentile calibrator class.

    Args:
        num_bins (int, optional): number of bins to create a new histogram
                                    for collecting tensor values. Defaults to 2048.
        percentile (float, optional): A float number between [0, 100]. Defaults to 99.999.
    """

    def __init__(self, num_bins=2048, percentile=99.999):
        """Initialize percentile calibrator class."""
        super(PercentileCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.percentile = percentile

    def collect_calib_data(self, datas):
        """Collect calibration range."""
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect_data(datas)
        self.compute_percentile_range(self.percentile)

    def compute_percentile_range(self, percentile):
        """Compute percentile range."""
        if percentile < 0 or percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        calib_hist, calib_bin_edges, min_range, max_range, th = self.collector.histogram
        total = calib_hist.sum()
        cdf = np.cumsum(calib_hist / total)
        percent_to_cut_one_side = (100.0 - percentile) / 200.0
        max_idx = np.searchsorted(cdf, 1.0 - percent_to_cut_one_side)
        min_idx = np.searchsorted(cdf, percent_to_cut_one_side)
        self._calib_min = calib_bin_edges[min_idx].astype("float32")
        self._calib_max = calib_bin_edges[max_idx].astype("float32")
        if self._calib_min < min_range:
            self._calib_min = min_range
        if self._calib_max > max_range:
            self._calib_max = max_range

    def clear(self):
        """Clear calibration range."""
        self._calib_min = None
        self._calib_max = None
        self.collector = None

    @property
    def method_name(self):
        """Get calibration method name."""
        return "percentile"


@calib_registry(calib_method="kl")
class KLCalibrator(CalibratorBase):
    """KL calibrator class.

    Args:
        num_bins (int, optional):number of bins to create a new histogram
                                    for collecting tensor values. Defaults to 128.
        num_quantized_bins (int, optional): number of quantized bins. Defaults to 128.
    """

    def __init__(self, num_bins=128, num_quantized_bins=128):
        """Initialize kl calibrator class."""
        super(KLCalibrator, self).__init__()
        self.collector = None
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins

    def collect_calib_data(self, datas):
        """Collect calibration range."""
        if not self.collector:
            self.collector = HistogramCollector(self.num_bins)
        self.collector.collect_data(datas)
        self.compute_kl_range()

    def compute_kl_range(self):
        """Compute kl range."""
        histogram = self.collector.histogram
        self._calib_min, self._calib_max = self.get_kl_threshold(histogram, self.num_quantized_bins)

    def get_kl_threshold(self, histogram, num_quantized_bins):
        """Compute kl threshold.

        Ref:
        https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
        https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py

        Args:
            histogram (tuple): hist, hist_edges, min, max and threshold
            num_quantized_bins (int): number of quantized bins.

        Returns:
            float: optimal threshold
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

    def clear(self):
        """Clear calibration range."""
        self._calib_min = None
        self._calib_max = None
        self.collector = None

    @property
    def method_name(self):
        """Get calibration method name."""
        return "kl"


class HistogramCollector:
    """Histogram collctor class."""

    def __init__(self, num_bins=2048):
        """Initialize histogram collctor."""
        self._num_bins = num_bins
        self._histogram = None

    def collect_data(self, datas):
        """Collect histogram data."""
        if isinstance(datas, list) and len(set([data.shape for data in datas])) != 1:
            for data in datas:
                if data.size == 0:  # pragma: no cover
                    continue
                self._collect_value(data)
        else:
            datas = np.asarray(datas)
            datas = datas.flatten()
            assert datas.size > 0, "collected intermediate data size" "should not be 0, please check augmented_model"
            self._collect_value(datas)

    def _collect_value(self, data):
        """Collect value."""
        data = np.asarray(data)
        min_range = np.min(data)
        max_range = np.max(data)

        th = max(abs(min_range), abs(max_range))
        if self._histogram is None:
            hist, hist_edges = np.histogram(data, self._num_bins, range=(-th, th))
            self._histogram = (hist, hist_edges, min_range, max_range, th)
        else:
            self._histogram = self.combine_histogram(self._histogram, data, min_range, max_range, th)

    def combine_histogram(self, old_hist, data_arr, new_min, new_max, new_th):
        """Combine histogram."""
        (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist

        if new_th <= old_th:
            hist, _ = np.histogram(data_arr, bins=len(old_hist), range=(-old_th, old_th))
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
                hist, hist_edges = np.histogram(data_arr, bins=new_num_bins, range=(-new_th, new_th))
                hist[half_increased_bins : new_num_bins - half_increased_bins] += old_hist
            return (
                hist,
                hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                new_th,
            )

    @property
    def histogram(self):
        """Get histogram."""
        return self._histogram


def smooth_distribution(p, eps=0.0001):
    """Smooth distribution.

    Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor
    and taking the corresponding amount off the non-zero values.
    Ref:
    http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py

    Args:
        p (array): distribution array
        eps (float, optional): a small probability. Defaults to 0.0001.

    Returns:
        array: smoothed distribution
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    if not n_nonzeros:
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
