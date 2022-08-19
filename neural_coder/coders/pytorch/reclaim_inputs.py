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
from ...utils.line_operation import (
    get_line_indent_level,
    is_eval_func_model_name,
    get_line_left_hand_side,
    single_line_comment_or_empty_line_detection
)

import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class ReclaimInputs(object):
    def __init__(self, list_model_def_instance):
        self.list_model_def_instance = list_model_def_instance
        
    def print_info(self):
        for i in self.list_model_def_instance:
            logger.debug(f"i.print_info(): {i.print_info()}")

    # collect file transformation info and register (store) in globals 
    # (i.e. which file to add which lines at which location)
    def register_transformation(self): 
        list_code = []
        for i in globals.list_code_path:
            list_code.append(open(i, 'r').read())

        for ins in self.list_model_def_instance:
            model_name = ins.model_name
            file_path = ins.file_path
            model_def_line_idx = ins.model_def_line_idx
            function_def_line_idx = ins.function_def_line_idx
            class_name = ins.class_name
            
            # transformation
            file_path_idx = globals.list_code_path.index(file_path)
            lines = list_code[file_path_idx].split('\n')
            line_idx = 0

            # search inference line in this file, and also input_name
            inference_line = ""
            input_name = ""
            for i in range(len(lines)):
                line = lines[i]
                is_eval_func, eval_func_type = is_eval_func_model_name(model_name, line)
                if is_eval_func and "[coder-enabled]" not in line:
                    inference_line = line
                    input_name = line[line.find("(")+1:line.find(")")].replace("*","")  # get "c" in "a = b(**c)"

            # if there is already a "input = xxx", then quit this function
            if input_name != "":
                for i in range(len(lines)):
                    line = lines[i]
                    if not single_line_comment_or_empty_line_detection(line):
                        if input_name in line and "=" in line and line.find(input_name) < line.find("="):
                            return
            
            # add the created lines for inputs
            if inference_line != "" and input_name != "":
                for i in range(len(lines)):
                    line = lines[i]
                    is_eval_func, eval_func_type = is_eval_func_model_name(model_name, line)
                    if is_eval_func and "[coder-enabled]" not in line:
                        indent_level = get_line_indent_level(line)
                        trans_insert_location = i
                        lines_to_insert = ""
                        lines_to_insert += " " * indent_level + "try:" + "\n"
                        lines_to_insert += " " * indent_level + "    " + input_name + " = " + input_name + "\n"
                        lines_to_insert += " " * indent_level + "except:" + "\n"
                        lines_to_insert += " " * indent_level + "    pass"

                        if file_path not in globals.list_trans_insert_modified_file:
                            globals.list_trans_insert_modified_file.append(file_path)
                            globals.list_trans_insert_location_idxs.append([trans_insert_location])
                            globals.list_trans_insert_number_insert_lines.append([lines_to_insert.count("\n") + 1])
                            globals.list_trans_insert_lines_to_insert.append([lines_to_insert])
                        else:
                            idx = globals.list_trans_insert_modified_file.index(file_path)
                            globals.list_trans_insert_location_idxs[idx].append(trans_insert_location)
                            globals.list_trans_insert_number_insert_lines[idx].append(lines_to_insert.count("\n") + 1)
                            globals.list_trans_insert_lines_to_insert[idx].append(lines_to_insert)
                
                    line_idx += 1
        
        logger.debug(f"globals.list_trans_insert_modified_file: {globals.list_trans_insert_modified_file}")
        logger.debug(f"globals.list_trans_insert_location_idxs: {globals.list_trans_insert_location_idxs}")
        logger.debug(f"globals.list_trans_insert_number_insert_lines: {globals.list_trans_insert_number_insert_lines}")
        logger.debug(f"globals.list_trans_insert_lines_to_insert: {globals.list_trans_insert_lines_to_insert}")
