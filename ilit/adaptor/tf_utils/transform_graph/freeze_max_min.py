#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

import numpy as np
import math
import functools
import logging

logger = logging.getLogger()


def parse_input_graph(input_graph_def):
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            logger.info('Duplicate node name {}'.format(node.name))
    return input_node_map


def get_valid_log(max_min_log):
    output = []

    target_lines = [
        i.strip() for i in max_min_log if i.strip().find(';') != -1
    ]
    for i in target_lines:
        semi_count = i.count(';')
        if semi_count == 2:
            output.append(i)
        elif semi_count % 2 != 0:
            logger.warn("Invalid line")
        else:
            loop_times = int(semi_count / 2)
            semi_index = [
                index for index, value in enumerate(i) if value == ";"
            ]
            for index in range(loop_times - 1):
                output.append(i[semi_index[index * 2]:semi_index[index * 2 +
                                                                 2]])
            output.append(i[semi_index[loop_times * 2 - 2]:])
    return output


def get_all_data(data_piece):
    return [
        int(i)
        for i in data_piece.replace('[', ' ').replace(']', ' ').split(' ')
        if i.strip()
    ]


def get_all_fp32_data(data_piece):
    return [
        float(i)
        for i in data_piece.replace('[', ' ').replace(']', ' ').split(' ')
        if i.strip()
    ]


def generic_scale(max_value_data, range_max, range_min):
    number_of_bits = 32
    number_of_steps = 1 << number_of_bits
    range_adjust = float(number_of_steps / (number_of_steps - 1))
    range_total = float((range_max - range_min) * range_adjust)
    range_scale = float(range_total / number_of_steps)
    lowest_quantized = -1 << 31
    offset_input = float(float(max_value_data) - lowest_quantized)
    range_min_rounded = float(
        round(range_min / float(range_scale)) * float(range_scale))
    result = float(range_min_rounded + (offset_input * range_scale))
    return result


def expand_quantized_bins(quantized_bins, reference_bins):
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
            avg_bin_ele = quantized_bins[idx] / (num_merged_bins - zero_count +
                                                 0.0)
        for idx1 in range(j_start, j_end):
            expanded_quantized_bins[
                idx1] = 0 if reference_bins[idx1] == 0 else avg_bin_ele
        j_start += num_merged_bins
        j_end += num_merged_bins
        if (idx + 1) == len(quantized_bins) - 1:
            j_end = len(reference_bins)
    return expanded_quantized_bins


def safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
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
                logger.fatal("Fatal error!, idx = " + str(idx) +
                             " qindex = 0! p_idx = " + str(p_idx))
            tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
            tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
    return (tmp_sum1 - tmp_sum2) / P_sum


def combine_histogram(old_hist, arr):
    """ Collect layer histogram for arr and combine it with old histogram.
    """
    new_max = np.max(arr)
    new_min = np.min(arr)
    new_th = max(abs(new_min), abs(new_max))
    (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr,
                               bins=len(old_hist),
                               range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min),
                max(old_max, new_max), old_th)
    else:
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr,
                                        bins=new_num_bins,
                                        range=(-new_th, new_th))
        hist[half_increased_bins:new_num_bins -
             half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max,
                                                             new_max), new_th)


def get_tensor_histogram(tensor_data, bins=2048):
    max_val = np.max(tensor_data)
    min_val = np.min(tensor_data)
    th = max(abs(min_val), abs(max_val))

    hist, hist_edeges = np.histogram(tensor_data, bins=2048, range=(-th, th))

    return (hist, hist_edeges, max_val, min_val, th)


def get_optimal_scaling_factor(tensor_details, num_quantized_bins=255):
    hist = tensor_details[0]
    hist_edeges = tensor_details[1]
    max_val = tensor_details[2]
    min_val = tensor_details[3]
    th = tensor_details[4]

    if min_val >= 0:
        ending_iter = 2047
        starting_iter = int(ending_iter * 0.7)
        min_range = min_val
    else:
        min_range = -th
        starting_iter = 0
        ending_iter = 2047
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
    bin_width = hist_edeges[1] - hist_edeges[0]
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
            if (idx + 1) == num_quantized_bins - 1:
                j_end = i
        candidate_distr_Q = expand_quantized_bins(candidate_distr_Q_quantized,
                                                  reference_distr_bins)
        P_sum = sum(reference_distr_P)
        Q_sum = sum(candidate_distr_Q)
        kl_divergence = safe_entropy(reference_distr_P, P_sum,
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
    return (min_kl_index + 0.5) * bin_width + min_range


def parse_requantization_ranges_kl_fp32(fp32_log, print_node_mapping):
    valid_lines = get_valid_log(fp32_log)
    kl_appendix = "__;__KL:"
    valid_data = [i for i in valid_lines if i.find(kl_appendix) != -1]

    single_keys_prefix = sorted(
        set([i.split(kl_appendix)[0] for i in valid_data]))
    result = {}
    for node_name in single_keys_prefix:
        content_str = node_name + kl_appendix
        content_set = []
        key_name = print_node_mapping[node_name[1:].split('__print')
                                      [0]] + '_eightbit_requant_range'
        for line in valid_data:
            if line.find(content_str) != -1:
                content_set.append(line.split(content_str)[-1])
            else:
                pass

        all_transformed_data = functools.reduce(lambda a, b: a + b,
                                                content_set)

        kl = get_optimal_scaling_factor(
            get_all_fp32_data(all_transformed_data))

        result[key_name] = kl
    return result


def parse_requantization_ranges_kl(log_path):
    valid_lines = get_valid_log(log_path)

    kl_appendix = "__;__KL:"
    min_postfix = "_min_output"
    max_postfix = "_max_output"
    valid_data = [i for i in valid_lines if i.find(kl_appendix) != -1]

    single_keys_prefix = sorted(
        set([i.split(kl_appendix)[0] for i in valid_data]))
    result = {}

    for node_name in single_keys_prefix:
        min_out_str = node_name + kl_appendix + min_postfix
        max_out_str = node_name + kl_appendix + max_postfix
        content_str = node_name + kl_appendix
        key_name = node_name[1:].split(
            '_quantized_conv__print')[0] + '_requant_range'
        min_value_set = []
        max_value_set = []
        content_set = []
        for line in valid_data:
            if line.find(min_out_str) != -1:
                min_value = line.split('[')[-1].split(']')[0]
                min_value_set.append(min_value)
            elif line.find(max_out_str) != -1:
                max_value = line.split('[')[-1].split(']')[0]
                max_value_set.append(max_value)

            elif line.find(content_str) != -1:
                content_set.append(line.split(content_str)[-1])
            else:
                pass

        all_transformed_data = []
        for index, min_range_value in enumerate(min_value_set):
            #  step 0 translate data
            max_range_value = float(max_value_set[index])
            min_range_value = float(min_range_value)
            cur_data = get_all_data(content_set[index])
            for i in cur_data:
                all_transformed_data.append(
                    generic_scale(i, max_range_value, min_range_value))

        kl = get_optimal_scaling_factor(all_transformed_data)
        result[key_name] = kl

    return result


def parse_requantization_ranges(max_min_log):
    """
    Parse the max_min log to get requantization values
    :param max_min_log: input min max log file
    :return: dict saved the result
    """
    print_suffix = "__print__"
    post_fix = "__requant_min_max"
    lines = get_valid_log(max_min_log)
    res = {}
    temp_min = {}
    temp_max = {}
    for i in lines:
        if i.find(print_suffix + ";" + post_fix) == -1:
            continue
        max_line_data = i.split(print_suffix + ";" + post_fix)[-1]
        min_value = max_line_data.split('][')[0].split('[')[1]
        max_value = max_line_data.split('][')[1].split(']')[0]
        name = i.split(';')[1].strip()[:-len(print_suffix)]
        if name not in temp_min:
            temp_min[name] = []
        if name not in temp_max:
            temp_max[name] = []

        temp_min[name].append(float(min_value))
        temp_max[name].append(float(max_value))

    for key in temp_min:
        target_min_index = int(round(len(temp_min[key]) * 0.05))
        if target_min_index < 0:
            target_min_index = 0
        if key not in res:
            res[key] = []
        res[key].append(sorted(temp_min[key])[target_min_index])
    for key in temp_max:
        target_max_index = int(round(len(temp_max[key]) * 0.95))
        if target_max_index > len(temp_max[key]) - 1:
            target_max_index = len(temp_max[key]) - 1
        res[key].append(sorted(temp_max[key])[target_max_index])

    return res


def parse_max_min_log(max_min_log, fetch_max=True):
    """
    Parse the max_ming log file
    :param max_min_log: max_min log file
    :param fetch_max: parse for freeze_max or not
    :return: get the node name and value mapping
    """
    print_suffix = "__print__"
    if fetch_max:
        postfix = "__max:"
    else:
        postfix = "__min:"

    lines = get_valid_log(max_min_log)

    res = {}
    temp = {}
    for i in lines:
        if i.find(print_suffix + ";" + postfix) == -1:
            continue
        max_line_data = i.split(';')
        name = max_line_data[1][:-len(print_suffix)]
        value = max_line_data[-1].split('[')[-1].split(']')[0]
        if "eightbit" in name and name not in temp:
            temp[name] = []
        if "eightbit" in name:
            temp[name].append(float(value))
    for key in temp:
        target_index = int(len(temp[key]) * 0.95)
        if target_index > len(temp[key]) - 1:
            target_index = len(temp[key]) - 1
        res[key] = sorted(temp[key])[target_index]
    return res


def generate_output_graph_ranges(input_node_map, range_info, device):
    output_graph_def = graph_pb2.GraphDef()
    inputs_to_rename = {}
    for node in input_node_map:
        if node in range_info:
            min_node = node_def_pb2.NodeDef()
            min_node.op = "HostConst" if device == "gpu" else "Const"
            min_node.name = node + "/frozen_min"
            inputs_to_rename[node + ":0"] = min_node.name + ":0"
            min_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            min_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    float(range_info[node][0]), dtypes.float32, [])))

            max_node = node_def_pb2.NodeDef()
            max_node.op = "HostConst" if device == "gpu" else "Const"
            max_node.name = node + "/frozen_max"
            inputs_to_rename[node + ":1"] = max_node.name + ":0"
            max_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            max_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    float(range_info[node][1]), dtypes.float32, [])))
            output_graph_def.node.extend([min_node, max_node])
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(input_node_map[node])
            output_graph_def.node.extend([new_node])

    for node in output_graph_def.node:
        found_index = []

        for input_index, input_name in enumerate(node.input):
            for _, v in enumerate(inputs_to_rename):
                if input_name == v:
                    found_index.append(input_index)

        if found_index:
            for sub_index in found_index:
                node.input[sub_index] = inputs_to_rename[node.input[sub_index]]

    return output_graph_def


def generate_output_graph(input_node_map, max_name_value, is_max=True):
    """
    Generate transformed graph for freeze_max/freeze_min transformation.
    :param input_node_map: input node name and nodedef mapping
    :param max_name_value: target values
    :param is_max: freeze_max flag
    :return: transformed graph
    """
    output_graph_def = graph_pb2.GraphDef()
    inputs_to_rename = {}
    for node in input_node_map:
        if node in max_name_value:
            new_node = node_def_pb2.NodeDef()
            new_node.op = "Const"
            new_node_postfix = "/frozen_max_only" if is_max else "/frozen_min_only"
            new_node.name = node + new_node_postfix
            inputs_to_rename[node] = new_node.name + ":0"
            new_node.attr["dtype"].CopyFrom(
                attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            new_node.attr["value"].CopyFrom(
                attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                    float(max_name_value[node]), dtypes.float32, [])))
        else:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(input_node_map[node])
        output_graph_def.node.extend([new_node])

    for node in output_graph_def.node:
        found = False
        found_index = -1
        found_value = ""
        for input_index, input_name in enumerate(node.input):
            for _, v in enumerate(inputs_to_rename):
                if input_name == v:
                    found = True
                    found_index = input_index
                    found_value = v
                    break
            if found:
                break
        if found:
            post_fix = '/frozen_max_only:0' if is_max else '/frozen_min_only:0'
            node.input[found_index] = found_value + post_fix

    return output_graph_def


def freeze_requantization_range(input_graph_def,
                                max_min_log,
                                tensor_histogram=None,
                                print_node_mapping=None, device='cpu'):
    """
    Freeze requantization range graph transformation
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log file
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    range_info = parse_requantization_ranges(max_min_log)
    if tensor_histogram:
        for key in tensor_histogram:
            kl_value = get_optimal_scaling_factor(tensor_histogram[key])
            if key in range_info:
                range_info[key][-1] = kl_value
                range_info[key][0] = 0

    return generate_output_graph_ranges(input_node_map, range_info, device)


def freeze_max(input_graph_def, max_min_log):
    """
    Freeze max graph transformation
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    max_name_value = parse_max_min_log(max_min_log, True)
    return generate_output_graph(input_node_map, max_name_value, True)


def freeze_min(input_graph_def, max_min_log):
    """
    Freeze min graph transformation.
    :param input_graph_def: input graphdef
    :param max_min_log: max_min_log file
    :return: transformed graph
    """
    input_node_map = parse_input_graph(input_graph_def)
    max_name_value = parse_max_min_log(max_min_log, False)
    return generate_output_graph(input_node_map, max_name_value, False)
