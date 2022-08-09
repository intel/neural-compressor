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


# whethre to consider all import modules and perform transformation based on all codes plus all import modules
import logging
consider_imports = True

# target batch size for feature of changing PyTorch batch size
target_batch_size = 1

# number of benchmark iteration for feature of PyTorch benchmark
num_benchmark_iteration = 30

# print info for debugging purpose
logging_level = logging.INFO


def reset_globals():
    global list_code_path

    global list_code_line_instance

    global list_class_def_instance
    global list_class_name
    global list_parent_class_name

    global list_model_def_instance
    global list_model_name

    global list_trans_insert_modified_file
    global list_trans_insert_location_idxs
    global list_trans_insert_number_insert_lines
    global list_trans_insert_lines_to_insert

    global list_trans_indenting_modified_file
    global list_trans_indenting_location_idxs
    global list_trans_indenting_level

    global list_all_function_name
    global list_all_function_return_item

    global list_wrapper_base_function_name
    global list_wrapper_children_function_name
    global list_wrapper_all_function_name

    list_code_path = []
    list_code_line_instance = []  # list of CodeLine instances

    list_class_def_instance = []  # list of ClassDefinition instances
    list_class_name = []  # list of class names
    list_parent_class_name = []

    list_model_def_instance = []
    list_model_name = []

    # for code transformation recording
    list_trans_insert_modified_file = []
    list_trans_insert_location_idxs = []
    list_trans_insert_number_insert_lines = []
    list_trans_insert_lines_to_insert = []

    list_trans_indenting_modified_file = []
    list_trans_indenting_location_idxs = []
    list_trans_indenting_level = []

    list_all_function_name = []
    list_all_function_return_item = []

    list_wrapper_base_function_name = []
    list_wrapper_children_function_name = []
    list_wrapper_all_function_name = []
