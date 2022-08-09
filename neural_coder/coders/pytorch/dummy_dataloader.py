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
from ...utils.line_operation import get_line_indent_level, is_eval_func_model_name, get_line_lhs

import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class DummyDataLoader(object):
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
            for i in range(len(lines)):  # each item is a str of this code line
                line = lines[i]
                if "DataLoader(" in line and "=" in line and line.find("=") < line.find("DataLoader"):
                    dataloader_name = get_line_lhs(line)
                    dataloader_def_line_idx = i

            if dataloader_name != "":
                return
            else:
                input_dimension_str = "3, 224, 224)"
                for i in range(len(lines)):
                    line = lines[i]
                    if ("input" in line and "=" in line and line.find("=") > line.find("input")) \
                        or ("image" in line and "=" in line and line.find("=") > line.find("image")):
                        input_dimension_str = line[line.find(",")+2:]

                for i in range(len(lines)):
                    line = lines[i]
                    if line_idx == model_def_line_idx:
                        indent_level = get_line_indent_level(line)
                        lines_to_insert = ""
                        lines_to_insert += " " * indent_level + "import torch" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "from torch.utils.data import Dataset" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "class DummyDataset(Dataset):" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "    def __init__(self, *shapes, num_samples: int = 10000):" + "\n"
                        lines_to_insert += " " * indent_level + "        super().__init__()" + "\n"
                        lines_to_insert += " " * indent_level + "        self.shapes = shapes" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "        self.num_samples = num_samples" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "    def __len__(self):" + "\n"
                        lines_to_insert += " " * indent_level + "        return self.num_samples" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "    def __getitem__(self, idx: int):" + "\n"
                        lines_to_insert += " " * indent_level + "        sample = []" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "        for shape in self.shapes:" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "            spl = torch.rand(*shape)" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "            sample.append(spl)" + "\n"
                        lines_to_insert += " " * indent_level + "        return sample" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "from torch.utils.data import DataLoader" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "my_dataset = DummyDataset((" + \
                            input_dimension_str + ", (1, ))" + "\n"
                        lines_to_insert += " " * indent_level + \
                            "my_dataloader = DataLoader(my_dataset, batch_size=1)"

                        trans_insert_location = 0

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
