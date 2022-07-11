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


from .... import globals
from ....utils.line_operation import get_line_indent_level, is_eval_func_model_name, get_line_lhs

import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class IPEX(object):
    def __init__(self, list_model_def_instance, mode):
        self.list_model_def_instance = list_model_def_instance
        self.mode = mode

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
            try:
                model_name = ins.model_name
                file_path = ins.file_path
                model_def_line_idx = ins.model_def_line_idx
                function_def_line_idx = ins.function_def_line_idx
                class_name = ins.class_name

                # transformation
                file_path_idx = globals.list_code_path.index(file_path)
                lines = list_code[file_path_idx].split('\n')
                line_idx = 0

                # search eval func and inputs
                eval_func_line = ""
                for i in range(len(lines)):
                    line = lines[i]
                    if is_eval_func_model_name(model_name, line):
                        eval_func_line = line
                        inputs = eval_func_line[eval_func_line.find(
                            "(")+1:eval_func_line.find(")")]

                # search dataloader or inputs definition line
                # (for getting line_idx to insert after it, because IPEX INT8 API contains the referral to inputs)
                dataloader_inputs_def_line_idx = 0
                for i in range(len(lines)):
                    line = lines[i]
                    if "DataLoader(" in line and "=" in line and line.find("=") < line.find("DataLoader"):
                        dataloader_inputs_def_line_idx = i
                    if inputs.replace("*", "") in line \
                            and "=" in line and line.find("=") > line.find(inputs.replace("*", "")):
                        dataloader_inputs_def_line_idx = i
                if dataloader_inputs_def_line_idx == 0:
                    logger.warning(
                        f"You must define an inputs/dataloader and have an evaluate funcion \
                            (e.g. 'output = model(input)') in this file: {file_path}")
                    continue

                for i in range(len(lines)):
                    line = lines[i]
                    if line_idx == model_def_line_idx:
                        indent_level = get_line_indent_level(line)
                        lines_to_insert = ""

                        if self.mode == "fp32":
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "import intel_extension_for_pytorch as ipex" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + model_name + ".eval()" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + model_name + \
                                " = ipex.optimize(" + model_name + \
                                ", dtype=torch.float32)"
                        if self.mode == "bf16":
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "import intel_extension_for_pytorch as ipex" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + model_name + ".eval()" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + model_name + \
                                " = ipex.optimize(" + model_name + \
                                ", dtype=torch.bfloat16)"
                        if self.mode == "int8_static_quant":
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "import intel_extension_for_pytorch as ipex" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "qconfig = ipex.quantization.default_static_qconfig" + "\n"
                            lines_to_insert += " " * indent_level + model_name + \
                                " = ipex.quantization.prepare(" + model_name + \
                                ", qconfig, example_inputs=" + inputs.replace(
                                    "*", "") + ", inplace=False)" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + \
                                " " * 4 + "for i in range(10):" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + \
                                eval_func_line.lstrip() + " # Neural Coder appended" + "\n"
                            lines_to_insert += " " * indent_level + model_name + \
                                " = ipex.quantization.convert(" + \
                                model_name + ")" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                eval_func_line.lstrip() + " # Neural Coder appended" + "\n"
                        if self.mode == "int8_dynamic_quant":
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "import intel_extension_for_pytorch as ipex" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "qconfig = ipex.quantization.default_dynamic_qconfig" + "\n"
                            lines_to_insert += " " * indent_level + model_name + \
                                " = ipex.quantization.prepare(" + model_name + \
                                ", qconfig, example_inputs=" + inputs.replace(
                                    "*", "") + ", inplace=False)" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + \
                                " " * 4 + "for i in range(10):" + "\n"
                            lines_to_insert += " " * indent_level + " " * 8 + \
                                eval_func_line.lstrip() + " # Neural Coder appended" + "\n"
                            lines_to_insert += " " * indent_level + model_name + \
                                " = ipex.quantization.convert(" + \
                                model_name + ")" + "\n"
                            lines_to_insert += " " * indent_level + "with torch.no_grad():" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                eval_func_line.lstrip() + " # Neural Coder appended" + "\n"

                        if ")" in line:  # e.g. model = Net(xxx)
                            if "int8" in self.mode:
                                trans_insert_location = max(
                                    line_idx + 1, dataloader_inputs_def_line_idx + 1)
                            else:
                                trans_insert_location = line_idx + 1
                        else:  # e.g. model = Net(xxx，
                            #                  xxx，
                            #                  xxx
                            #      )
                            do_search = True
                            i_search = 1
                            while do_search:
                                following_line = lines[line_idx + i_search]
                                if ")" in following_line and following_line[indent_level] == ")":
                                    do_search = False
                                i_search += 1
                            if "int8" in self.mode:
                                trans_insert_location = max(
                                    line_idx + i_search, dataloader_inputs_def_line_idx + 1)
                            else:
                                trans_insert_location = line_idx + i_search

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
            f"globals.list_trans_insert_modified_file: {globals.list_trans_insert_modified_file}")
        logger.debug(
            f"globals.list_trans_insert_location_idxs: {globals.list_trans_insert_location_idxs}")
        logger.debug(
            f"globals.list_trans_insert_number_insert_lines: {globals.list_trans_insert_number_insert_lines}")
        logger.debug(
            f"globals.list_trans_insert_lines_to_insert: {globals.list_trans_insert_lines_to_insert}")
