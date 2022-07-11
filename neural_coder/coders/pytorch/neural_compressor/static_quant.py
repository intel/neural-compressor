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


class StaticQuant(object):
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

            # search DataLoader
            dataloader_name = ""
            for i in range(len(lines)):
                line = lines[i]
                if "DataLoader(" in line and "=" in line and line.find("=") < line.find("DataLoader"):
                    dataloader_name = get_line_lhs(line)
                    dataloader_def_line_idx = i

            # search eval func
            eval_func_line = ""
            for i in range(len(lines)):
                line = lines[i]
                if is_eval_func_model_name(model_name, line):
                    eval_func_line = line

            # for scripts that does not have "dataloader" defined but does have "input" defined
            if dataloader_name == "":
                dataloader_name = "my_dataloader"
                dataloader_def_line_idx = 0
                for i in range(len(lines)):
                    line = lines[i]
                    if ("input" in line and "=" in line and line.find("=") > line.find("input")) \
                        or ("image" in line and "=" in line and line.find("=") > line.find("image")):
                        # for adding the static quant after input/image definition line
                        dataloader_def_line_idx = i

            if eval_func_line != "":
                for i in range(len(lines)):
                    line = lines[i]
                    if line_idx == model_def_line_idx:
                        indent_level = get_line_indent_level(line)
                        lines_to_insert = ""
                        lines_to_insert += " " * indent_level + \
                            "def eval_func(" + model_name + "):" + \
                                           " # NeuralCoder: pytorch_inc_static_quant [start line]" + "\n"
                        lines_to_insert += " " * indent_level + " " * 4 + \
                            eval_func_line.lstrip() + " # Neural Coder appended" + "\n"
                        lines_to_insert += " " * indent_level + " " * 4 + "return 1" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "from neural_compressor.conf.config import QuantConf" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "from neural_compressor.experimental import Quantization, common" + "\n"
                        lines_to_insert += " " * indent_level + "quant_config = QuantConf()" + "\n"
                        lines_to_insert += " " * indent_level + \
                            'quant_config.usr_cfg.model.framework = "pytorch_fx"' + "\n"
                        lines_to_insert += " " * indent_level + \
                            "quantizer = Quantization(quant_config)" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "quantizer.model = common.Model(" + \
                            model_name + ")" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "quantizer.calib_dataloader = " + dataloader_name + "\n"
                        lines_to_insert += " " * indent_level + "quantizer.eval_func = eval_func" + "\n"
                        lines_to_insert += " " * indent_level + model_name + " = quantizer()" + "\n"
                        lines_to_insert += " " * indent_level + model_name + " = " + model_name + \
                            ".model" + \
                            " # NeuralCoder: pytorch_inc_static_quant [end line]"

                        if ")" in line:  # e.g. model = Net(xxx)
                            # in case dataloader is defined under model, has to insert after dataloader definition
                            # because static quant API refers to the dataloader_name
                            trans_insert_location = max(
                                line_idx + 1, dataloader_def_line_idx + 1)
                        else:  # e.g. model = Net(xxx，
                            #                  xxx，
                            #                  xxx
                            #      )
                            do_search = True
                            i_search = 1
                            while do_search:
                                following_line = lines[line_idx + i_search]
                                if ")" in following_line:
                                    do_search = False
                                i_search += 1
                            # in case dataloader is defined under model, has to insert after dataloader definition
                            # because static quant API refers to the dataloader_name
                            trans_insert_location = max(
                                line_idx + i_search, dataloader_def_line_idx + 1)

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

        logger.debug(
            f"globals.list_trans_insert_modified_file: {globals.list_trans_insert_modified_file}")
        logger.debug(
            f"globals.list_trans_insert_location_idxs: {globals.list_trans_insert_location_idxs}")
        logger.debug(
            f"globals.list_trans_insert_number_insert_lines: {globals.list_trans_insert_number_insert_lines}")
        logger.debug(
            f"globals.list_trans_insert_lines_to_insert: {globals.list_trans_insert_lines_to_insert}")
