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


from ... import globals
from ...utils.line_operation import get_line_indent_level, is_eval_func_model_name

import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class Benchmark(object):
    def __init__(self):
        pass

    # collect file transformation info and register (store) in globals
    # (i.e. which file to add which lines at which location)
    def register_transformation(self):
        for file_path in globals.list_code_path:
            try:
                code = open(file_path, 'r').read()
                lines = code.split('\n')
                line_idx = 0
                for i in range(len(lines)):
                    line = lines[i]
                    for model_name in globals.list_model_name:
                        if is_eval_func_model_name(model_name, line) and "# Neural Coder appended" not in line:
                            indent_level = get_line_indent_level(line)

                            # 1. indenting
                            # indenting can have multiple location, so is a list of numbers
                            trans_indenting_location = []
                            trans_indenting_level = []

                            if ")" in line:  # e.g. model(xxx)
                                trans_indenting_location.append(line_idx)
                                trans_indenting_level.append(1)
                            else:  # e.g. model(xxx，
                                #            xxx，
                                #            xxx
                                #      )
                                trans_indenting_location.append(line_idx)
                                trans_indenting_level.append(1)
                                do_search = True
                                i_search = 1
                                while do_search:
                                    trans_indenting_location.append(
                                        line_idx + i_search)
                                    trans_indenting_level.append(1)
                                    following_line = lines[line_idx + i_search]
                                    if ")" in following_line:
                                        do_search = False
                                    i_search += 1

                            # 1. register indenting: transform "model(input)" to "    model(input)"
                            if file_path not in globals.list_trans_indenting_modified_file:
                                globals.list_trans_indenting_modified_file.append(
                                    file_path)
                                globals.list_trans_indenting_location_idxs.append(
                                    trans_indenting_location)
                                globals.list_trans_indenting_level.append(
                                    trans_indenting_level)
                            else:
                                idx = globals.list_trans_indenting_modified_file.index(
                                    file_path)
                                for i in trans_indenting_location:
                                    globals.list_trans_indenting_location_idxs[idx].append(
                                        i)
                                for i in trans_indenting_level:
                                    globals.list_trans_indenting_level[idx].append(
                                        i)

                            # 2-1. insert (before model(input))
                            trans_insert_location = []  # insert only has 1 location, so is a number

                            trans_insert_location = line_idx  # insert before

                            lines_to_insert = ""
                            lines_to_insert += " " * indent_level + "import time" + "\n"
                            lines_to_insert += " " * indent_level + "count_iter_ = 0" + "\n"
                            lines_to_insert += " " * indent_level + "total_time_ = 0" + "\n"
                            lines_to_insert += " " * indent_level + "num_iter_ = " + \
                                globals.num_benchmark_iteration + "\n"
                            lines_to_insert += " " * indent_level + "num_warmup_iter_ = 10" + "\n"
                            lines_to_insert += " " * indent_level + "list_batch_time_ = []" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "for i_ in range(num_iter_):" + "\n"
                            lines_to_insert += " " * indent_level + " " * \
                                4 + "count_iter_ = count_iter_ + 1" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                "if count_iter_ > num_warmup_iter_:" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + "t1_ = time.time()"

                            # 2-1. register insert (before model(input))
                            if file_path not in globals.list_trans_insert_modified_file:
                                globals.list_trans_insert_modified_file.append(
                                    file_path)
                                globals.list_trans_insert_location_idxs.append(
                                    [trans_insert_location])
                                globals.list_trans_insert_number_insert_lines.append(
                                    [lines_to_insert.count("\n") + 1])
                                globals.list_trans_insert_lines_to_insert.append(
                                    [lines_to_insert])
                            else:
                                idx = globals.list_trans_insert_modified_file.index(
                                    file_path)
                                globals.list_trans_insert_location_idxs[idx].append(
                                    trans_insert_location)
                                globals.list_trans_insert_number_insert_lines[idx].append(
                                    lines_to_insert.count("\n") + 1)
                                globals.list_trans_insert_lines_to_insert[idx].append(
                                    lines_to_insert)

                            # 2-2. insert (after model(input))
                            trans_insert_location = []  # insert only has 1 location, so is a number

                            if ")" in line:  # e.g. model()
                                trans_insert_location = line_idx + 1  # insert after
                            else:  # e.g. model(xxx，
                                #              xxx，
                                #              xxx
                                #        )
                                do_search = True
                                i_search = 1
                                while do_search:
                                    following_line = lines[line_idx + i_search]
                                    if ")" in following_line and following_line[indent_level] == ")":
                                        do_search = False
                                    i_search += 1
                                trans_insert_location = line_idx + i_search  # insert after

                            lines_to_insert = ""
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                "if count_iter_ > num_warmup_iter_:" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + "t2_ = time.time()" + "\n"
                            lines_to_insert += " " * indent_level + \
                                " " * 8 + "batch_time_ = t2_ - t1_" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + \
                                "list_batch_time_.append(batch_time_)" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + \
                                "total_time_ = total_time_ + batch_time_" + "\n"
                            lines_to_insert += " " * indent_level + \
                                'print("Neural_Coder_Bench_IPS: ",' \
                                'round((num_iter_ - num_warmup_iter_) / total_time_, 3))' + "\n"
                            lines_to_insert += " " * indent_level + \
                                'print("Neural_Coder_Bench_MSPI: ",' \
                                'round(total_time_ / (num_iter_ - num_warmup_iter_) * 1000, 3))' + "\n"
                            lines_to_insert += " " * indent_level + "list_batch_time_.sort()" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "p50_latency_ = list_batch_time_[int(len(list_batch_time_) * 0.50) - 1] * 1000" \
                                + "\n"
                            lines_to_insert += " " * indent_level + \
                                "p90_latency_ = list_batch_time_[int(len(list_batch_time_) * 0.90) - 1] * 1000" \
                                + "\n"
                            lines_to_insert += " " * indent_level + \
                                "p99_latency_ = list_batch_time_[int(len(list_batch_time_) * 0.99) - 1] * 1000" \
                                + "\n"
                            lines_to_insert += " " * indent_level + \
                                'print("Neural_Coder_Bench_P50: ", round(p50_latency_, 3))' + "\n"
                            lines_to_insert += " " * indent_level + \
                                'print("Neural_Coder_Bench_P90: ", round(p90_latency_, 3))' + "\n"
                            lines_to_insert += " " * indent_level + \
                                'print("Neural_Coder_Bench_P99: ", round(p99_latency_, 3))' + "\n"

                            # 2-2. register insert (after model(input))
                            if file_path not in globals.list_trans_insert_modified_file:
                                globals.list_trans_insert_modified_file.append(
                                    file_path)
                                globals.list_trans_insert_location_idxs.append(
                                    [trans_insert_location])
                                globals.list_trans_insert_number_insert_lines.append(
                                    [lines_to_insert.count("\n") + 1])
                                globals.list_trans_insert_lines_to_insert.append(
                                    [lines_to_insert])
                            else:
                                idx = globals.list_trans_insert_modified_file.index(
                                    file_path)
                                globals.list_trans_insert_location_idxs[idx].append(
                                    trans_insert_location)
                                globals.list_trans_insert_number_insert_lines[idx].append(
                                    lines_to_insert.count("\n") + 1)
                                globals.list_trans_insert_lines_to_insert[idx].append(
                                    lines_to_insert)

                    line_idx += 1
            except:
                logger.debug(
                    f"This file has skipped patching due to unrecognizable code format: {file_path}")

        logger.debug(
            f"globals.list_trans_indenting_modified_file: {globals.list_trans_indenting_modified_file}")
        logger.debug(
            f"globals.list_trans_indenting_location_idxs: {globals.list_trans_indenting_location_idxs}")
        logger.debug(
            f"globals.list_trans_indenting_level: {globals.list_trans_indenting_level}")

        logger.debug(
            f"globals.list_trans_insert_modified_file: {globals.list_trans_insert_modified_file}")
        logger.debug(
            f"globals.list_trans_insert_location_idxs: {globals.list_trans_insert_location_idxs}")
        logger.debug(
            f"globals.list_trans_insert_number_insert_lines: {globals.list_trans_insert_number_insert_lines}")
        logger.debug(
            f"globals.list_trans_insert_lines_to_insert: {globals.list_trans_insert_lines_to_insert}")
