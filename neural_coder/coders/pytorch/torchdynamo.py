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


class TorchDynamo(object):
    def __init__(self):
        pass

    # collect file transformation info and register (store) in globals
    # (i.e. which file to add which lines at which location)
    def register_transformation(self):
        for file_path in globals.list_code_path:
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

                        # 2. insert
                        trans_insert_location = line_idx  # insert only has 1 location, so is a number

                        lines_to_insert = ""
                        lines_to_insert += " " * indent_level + "import torchdynamo" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "with torchdynamo.optimize(dynamo_backend):"

                        # 1. indenting: transform "model(input)" to "    model(input)"
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

                        # 2. insert: add "with autocast()" line
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


class TorchDynamoJITScript(object):
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
                for i in range(len(lines)):
                    line = lines[i]
                    if line_idx == model_def_line_idx:
                        indent_level = get_line_indent_level(line)
                        lines_to_insert = ""

                        if self.mode == "plain":
                            lines_to_insert += " " * indent_level + "from typing import List" + "\n"
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + "import torchdynamo" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "def dynamo_backend(gm: torch.fx.GraphModule," \
                                "example_inputs: List[torch.Tensor]):" + "\n"
                            lines_to_insert += " " * indent_level + \
                                " " * 4 + "return torch.jit.script(gm)"
                        if self.mode == "ofi":
                            lines_to_insert += " " * indent_level + "from typing import List" + "\n"
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + "import torchdynamo" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "def dynamo_backend(gm: torch.fx.GraphModule," \
                                "example_inputs: List[torch.Tensor]):" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                "return torch.jit.optimize_for_inference(torch.jit.script(gm))"

                        if ")" in line:  # e.g. model = Net(xxx)
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


class TorchDynamoJITTrace(object):
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
                for i in range(len(lines)):
                    line = lines[i]
                    if line_idx == model_def_line_idx:
                        indent_level = get_line_indent_level(line)
                        lines_to_insert = ""

                        if self.mode == "plain":
                            lines_to_insert += " " * indent_level + "from typing import List" + "\n"
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + "import torchdynamo" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "def dynamo_backend(gm: torch.fx.GraphModule, \
                                    example_inputs: List[torch.Tensor]):" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                "return torch.jit.trace(gm, example_inputs)"
                        if self.mode == "ofi":
                            lines_to_insert += " " * indent_level + "from typing import List" + "\n"
                            lines_to_insert += " " * indent_level + "import torch" + "\n"
                            lines_to_insert += " " * indent_level + "import torchdynamo" + "\n"
                            lines_to_insert += " " * indent_level + \
                                "def dynamo_backend(gm: torch.fx.GraphModule, \
                                    example_inputs: List[torch.Tensor]):" + "\n"
                            lines_to_insert += " " * indent_level + " " * 4 + \
                                "return torch.jit.optimize_for_inference(torch.jit.trace(gm, example_inputs))"

                        if ")" in line:  # e.g. model = Net(xxx)
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
