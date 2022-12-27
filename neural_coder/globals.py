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

import logging

# whethre to consider all import modules and perform transformation based on all codes plus all import modules
consider_imports = True

# target batch size for feature of changing PyTorch batch size
target_batch_size = 1

# mark for successful batch size change
batch_size_changed = False

# number of benchmark iteration for feature of PyTorch benchmark
num_benchmark_iteration = 30

# print info for debugging purpose
logging_level = logging.INFO

# print code line info for debug use
print_code_line_info = False

# load transformers class def by a cache file instead of on-the-fly catch
cache_load_transformers = True

# detected device
device = "cpu_with_amx"

# device compatibility of the code: e.g. ["cpu", "cuda"], ["cuda"]
list_code_device_compatibility = ["cuda"]

# quantization config for HuggingFace optimum-intel optimizations
# it is either "" (None) or "xxx" (a string of config path)
optimum_quant_config = ""

# code domain
code_domain = ""

# modular design
use_modular = True
modular_item = "" # str

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

    global list_trans_indent_modified_file
    global list_trans_indent_location_idxs
    global list_trans_indent_level

    global list_all_function_name
    global list_all_function_return_item

    global list_wrapper_base_function_name
    global list_wrapper_children_function_name
    global list_wrapper_all_function_name

    global list_calib_dataloader_name
    global list_eval_func_lines
    global list_eval_func_name

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

    list_trans_indent_modified_file = []
    list_trans_indent_location_idxs = []
    list_trans_indent_level = []

    list_all_function_name = []
    list_all_function_return_item = []

    list_wrapper_base_function_name = []
    list_wrapper_children_function_name = []
    list_wrapper_all_function_name = []

    list_calib_dataloader_name = []
    list_eval_func_lines = []
    list_eval_func_name = []
