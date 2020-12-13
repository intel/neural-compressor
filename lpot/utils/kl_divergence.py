#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

import math


class KL_Divergence(object):
    '''The class of supporting KL divergence calibration algorithm.

       Args:
           TODO:
    '''

    def __init__(self):
        pass

    def expand_quantized_bins(self, quantized_bins, reference_bins):
        """expand quantized bins"""
        expanded_quantized_bins = [0] * len(reference_bins)
        num_merged_bins = int(len(reference_bins) / len(quantized_bins))
        j_start = 0
        j_end = num_merged_bins
        for idx in range(len(quantized_bins)):
            zero_count = reference_bins[j_start:j_end].count(0)
            num_merged_bins = j_end - j_start
            if zero_count == num_merged_bins:
                avg_bin_ele = 0
            else:
                avg_bin_ele = quantized_bins[idx] / (num_merged_bins -
                                                     zero_count + 0.0)
            for idx1 in range(j_start, j_end):
                expanded_quantized_bins[
                    idx1] = 0 if reference_bins[idx1] == 0 else avg_bin_ele
            j_start += num_merged_bins
            j_end += num_merged_bins
            if idx + 1 == len(quantized_bins) - 1:
                j_end = len(reference_bins)
        return expanded_quantized_bins

    def safe_entropy(self, reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
        """ safe entropy """
        assert len(reference_distr_P) == len(candidate_distr_Q)
        tmp_sum1 = 0
        tmp_sum2 = 0
        for idx in range(len(reference_distr_P)):
            p_idx = reference_distr_P[idx]
            q_idx = candidate_distr_Q[idx]
            if p_idx == 0:
                tmp_sum1 += 0
                tmp_sum2 += 0
            else:
                if q_idx == 0:
                    print("Fatal error!, idx = " + str(idx) +
                          " qindex = 0! p_idx = " + str(p_idx))
                tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
                tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
        return (tmp_sum1 - tmp_sum2) / P_sum

    def get_threshold(self,
                      hist,
                      hist_edges,
                      min_val,
                      max_val,
                      num_bins,
                      quantized_type,
                      num_quantized_bins=255):
        '''The interface of getting threshold per KL divergency algorithm.

           Args:
               historgram (tensor list): The tensor numpy array list including all ops
                                         in each iteration.
               quantized_type (string): string being "int8" or "uint8".
               number_bins (integer): number of bins used in KL.

           Return:
               threshold per quantized ops, it's scalar.
        '''

        if min_val >= 0:
            ending_iter = num_bins - 1
            starting_iter = int(ending_iter * 0.7)
        else:
            th = max(abs(max_val), abs(min_val))
            starting_iter = 0
            ending_iter = num_bins - 1
            if abs(max_val) > abs(min_val):
                while starting_iter < ending_iter:
                    if hist[starting_iter] == 0:
                        starting_iter += 1
                        continue
                    else:
                        break
                starting_iter += int((ending_iter - starting_iter) * 0.6)
            else:
                while ending_iter > 0:
                    if hist[ending_iter] == 0:
                        ending_iter -= 1
                        continue
                    else:
                        break
                starting_iter = int(0.6 * ending_iter)

        bin_width = hist_edges[1] - hist_edges[0]
        min_kl_divergence = 0
        min_kl_index = 0
        kl_inited = False

        for i in range(starting_iter, ending_iter + 1):
            reference_distr_P = hist[0:i].tolist()
            outliers_count = sum(hist[i:2048])
            if reference_distr_P[i - 1] == 0:
                continue
            reference_distr_P[i - 1] += outliers_count
            reference_distr_bins = reference_distr_P[:]
            candidate_distr_Q = hist[0:i].tolist()
            num_merged_bins = int(i / num_quantized_bins)
            candidate_distr_Q_quantized = [0] * num_quantized_bins
            j_start = 0
            j_end = num_merged_bins

            for idx in range(num_quantized_bins):
                candidate_distr_Q_quantized[idx] = sum(
                    candidate_distr_Q[j_start:j_end])
                j_start += num_merged_bins
                j_end += num_merged_bins
                if idx + 1 == num_quantized_bins - 1:
                    j_end = i
            candidate_distr_Q = self.expand_quantized_bins(
                candidate_distr_Q_quantized, reference_distr_bins)
            P_sum = sum(reference_distr_P)
            Q_sum = sum(candidate_distr_Q)
            kl_divergence = self.safe_entropy(reference_distr_P, P_sum,
                                              candidate_distr_Q, Q_sum)
            if not kl_inited:
                min_kl_divergence = kl_divergence
                min_kl_index = i
                kl_inited = True
            elif kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                min_kl_index = i
            else:
                pass

        if min_kl_index == 0:
            while starting_iter > 0:
                if hist[starting_iter] == 0:
                    starting_iter -= 1
                    continue
                else:
                    break
            min_kl_index = starting_iter
        return (min_kl_index + 0.5) * bin_width
