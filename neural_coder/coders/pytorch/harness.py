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
    get_line_wo_comment,
    single_line_comment_or_empty_line_detection
)

import logging
import yaml
import sys
import os

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class Harness(object):
    def __init__(self, backend):
        self.backend = backend

    def print_info(self):
        for i in globals.list_model_def_instance:
            logger.debug(f"i.print_info(): {i.print_info()}")

    # collect file transformation info and register in globals
    # (i.e. which file to add which lines at which location)
    def register_transformation(self):
        backend_file = open(os.path.dirname(__file__) +
                            "/../../backends/" + self.backend + ".yaml")
        backend_dict = yaml.load(backend_file, Loader=yaml.BaseLoader)
        logger.debug(f"backend_dict: {backend_dict}")

        bk_trans_location = backend_dict["transformation"]["location"]  # string
        bk_trans_content = backend_dict["transformation"]["content"]  # string
        bk_trans_order = backend_dict["transformation"]["order"]  # list

        list_code = []
        for i in globals.list_code_path:
            list_code.append(open(i, 'r').read())

        for loc in bk_trans_location:

            # PART 1 - "model_definition_line"
            if "insert_below_model_definition_line" in loc:

                for ins in globals.list_model_def_instance:
                    model_name = ins.model_name
                    file_path = ins.file_path
                    model_def_line_idx = ins.model_def_line_idx

                    file_path_idx = globals.list_code_path.index(file_path)
                    lines = list_code[file_path_idx].split('\n')
                    line_idx = 0

                    # to check if this model has an inference line is in the file
                    # if not, skip this model
                    to_transform = False
                    for i in range(len(lines)):
                        line = lines[i]
                        if model_name + "(" in line or \
                            (model_name + "." in line and line.find(model_name) < line.find(".") and "(" in line):
                            to_transform = True
                    if not to_transform:
                        continue

                    ### information

                    # search DataLoader definition in this file
                    dataloader_name = ""
                    for i in range(len(lines)):
                        line = lines[i]
                        if not single_line_comment_or_empty_line_detection(line):
                            if ("DataLoader(" in line and "=" in line and line.find("=") < line.find("DataLoader")) \
                                    or ("dataloader" in line and "=" in line and \
                                        line.find("=") > line.find("dataloader")):
                                dataloader_def_line_indent_level = get_line_indent_level(line)
                                dataloader_name = get_line_left_hand_side(line)
                                dataloader_def_line_idx = i

                    # search inference line in this file, and also input_name
                    inference_line = ""
                    input_name = ""
                    for i in range(len(lines)):
                        line = lines[i]
                        is_eval_func, eval_func_type = is_eval_func_model_name(model_name, line)
                        if not single_line_comment_or_empty_line_detection(line):
                            if is_eval_func and "[coder-enabled]" not in line:
                                inference_line = line
                                input_name = line[line.find("(")+1:line.find(")")].replace("*","")
                                # get "c" in "a = b(**c)"

                    # search input definition in this file (if any)
                    if input_name != "":
                        for i in range(len(lines)):
                            line = lines[i]
                            if not single_line_comment_or_empty_line_detection(line):
                                if input_name in line and "=" in line and line.find("=") > line.find(input_name):
                                    input_def_line_indent_level = get_line_indent_level(line)
                                    input_def_line_idx = i
                    
                    # search trainer definition in this file (for transformers trainer only)
                    trainer_def_line_idx = -1
                    for i in range(len(lines)):
                        line = lines[i]
                        if not single_line_comment_or_empty_line_detection(line):
                            if "trainer = Trainer(" in line:
                                trainer_def_line_indent_level = get_line_indent_level(line)
                                trainer_def_line_idx = i

                    # serach model definition line and its end line index
                    # (only has 1 model definition line, because it's in loop of globals.list_model_def_instance)
                    for i in range(len(lines)):
                        line = lines[i]
                        if line_idx == model_def_line_idx and "[coder-enabled]" not in line:
                            model_def_line_indent_level = get_line_indent_level(line)
                            if ")" in line and line.count(")") == line.count("("):  # e.g. model = Net(xxx)
                                model_definition_end_line_idx = line_idx + 1
                            else:  # e.g. model = Net(xxx, \n xxx, \n xxx)
                                do_search = True
                                i_search = 1
                                while do_search:
                                    following_line = lines[line_idx + i_search]
                                    if ")" in following_line \
                                        and following_line.count(")") > following_line.count("("):
                                        do_search = False
                                    i_search += 1
                                model_definition_end_line_idx = line_idx + i_search
                        line_idx += 1

                    ### check

                    bk_trans_content_this = bk_trans_content[bk_trans_location.index(loc)]

                    if ("INPUT_NAME" in bk_trans_content_this and input_name == "") \
                            or ("DATALOADER_NAME" in bk_trans_content_this and dataloader_name == "") \
                            or ("INFERENCE_LINE" in bk_trans_content_this and inference_line == ""):
                        logger.info(f"Skipped due to not having enough information required by "
                                    "the transformation content specified in the config file "
                                    "(e.g. INPUT_NAME, DATALOADER_NAME, INFERENCE_LINE). "
                                    f"File path: {file_path}")
                        continue

                    ### location

                    # search for features to put below them
                    '''
                    Example (psuedo-code):
                    model = Net()
                    # jit script begin mark
                    model = torch.jit.script(model)
                    # jit script end mark (feature name + model name to handle multi-model situation)
                    model = ipex.optimize(model, "fp32") # "ipex fp32" must be put below "jit script"
                    '''
                    put_below_idx = 0
                    for i in range(len(lines)):
                        for item in bk_trans_order[0]["below"]:
                            line = lines[i]
                            if item in line and model_name in line:
                                put_below_idx = max(put_below_idx, i + 1)

                    # search for features to put above them
                    put_above_idx = sys.maxsize
                    for i in range(len(lines)):
                        for item in bk_trans_order[0]["above"]:
                            line = lines[i]
                            if item in line and model_name in line:
                                put_above_idx = min(put_above_idx, i)
                    
                    # location assignment (below model def / dataloader def / input def)
                    if "insert_below_model_definition_line" in loc:
                        trans_insert_location = \
                            min(max(model_definition_end_line_idx,
                                put_below_idx), put_above_idx)
                        if trainer_def_line_idx > 0:
                            trans_insert_location = trainer_def_line_idx - 1
                            # for transformers trainer to put right above trainer def
                    if "insert_below_dataloader_definition_line" in loc:
                        try:
                            dataloader_def_line_idx
                        except:
                            logger.warning(f"Skipped due to not having dataloader definition required by "
                                            "the transformation content specified in the config file. "
                                            f"File path: {file_path}")
                            continue
                        trans_insert_location = max(trans_insert_location,
                                                    min(max(dataloader_def_line_idx + 1,
                                                            put_below_idx), put_above_idx))
                    if "insert_below_input_definition_line" in loc:
                        try:
                            input_def_line_idx
                        except:
                            logger.warning(f"Skipped due to not having input definition required by "
                                            "the transformation content specified in the config file. "
                                            f"File path: {file_path}")
                            continue
                        trans_insert_location = max(trans_insert_location,
                                                    min(max(input_def_line_idx + 1,
                                                            put_below_idx), put_above_idx))
                    
                    insert_indent_level = get_line_indent_level(lines[trans_insert_location - 1])
                    if trainer_def_line_idx > 0: # for transformers trainer to put right above trainer def
                        insert_indent_level = get_line_indent_level(lines[trans_insert_location])
                    ### content

                    # lines to insert
                    lines_to_insert = bk_trans_content_this
                    # replace [+] indication with empty
                    lines_to_insert = lines_to_insert.replace(
                        "[+] ", " " * insert_indent_level)
                    # add begin indicator
                    lines_to_insert = " " * insert_indent_level + "# [NeuralCoder] " + \
                        self.backend + " for " + model_name + " [Beginning Line]\n" + lines_to_insert
                    # replace INDICATIONS with real stuff
                    lines_to_insert = lines_to_insert \
                        .replace("MODEL_NAME", model_name) \
                        .replace("INPUT_NAME", input_name) \
                        .replace("DATALOADER_NAME", dataloader_name) \
                        .replace("INFERENCE_LINE", inference_line.strip()) \
                        .replace("\n", " # [coder-enabled]\n")
                    # add end indicator
                    lines_to_insert += " # [coder-enabled]\n" + \
                        " " * insert_indent_level + "# [NeuralCoder] " + self.backend + " for " + \
                        model_name + " [Ending Line] # [coder-enabled]"

                    ### register

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

            # PART 2 - "inference line"
            if "indent_inference_line" in loc or \
                "insert_above_inference_line" in loc or \
                "insert_below_inference_line" in loc:

                for file_path in globals.list_code_path:
                    code = open(file_path, 'r').read()
                    lines = code.split('\n')
                    line_idx = 0
                    for i in range(len(lines)):
                        line = lines[i]
                        for model_name in globals.list_model_name:
                            is_eval_func, eval_func_type = is_eval_func_model_name(model_name, line)
                            if is_eval_func and "[coder-enabled]" not in line:
                                if eval_func_type == "non-forward":
                                    pass # do something
                                inference_line = line
                                inference_line_indent_level = get_line_indent_level(line)

                                if "indent_inference_line" in loc:
                                    bk_trans_content_this = bk_trans_content[bk_trans_location.index(loc)]
                                    add_indent_level = int(bk_trans_content_this)

                                    trans_indent_location = []
                                    # indent can have multiple location, so is a list of numbers
                                    trans_indent_level = []

                                    if ")" in line:  # e.g. model = Net(xxx)
                                        trans_indent_location.append(line_idx)
                                        trans_indent_level.append(add_indent_level)
                                    else:  # e.g. model = Net(xxx, \n xxx, \n xxx)
                                        trans_indent_location.append(line_idx)
                                        trans_indent_level.append(add_indent_level)
                                        do_search = True
                                        i_search = 1
                                        while do_search:
                                            trans_indent_location.append(line_idx + i_search)
                                            trans_indent_level.append(add_indent_level)
                                            following_line = lines[line_idx + i_search]
                                            if ")" in following_line:
                                                do_search = False
                                            i_search += 1

                                    ### register

                                    if file_path not in globals.list_trans_indent_modified_file:
                                        globals.list_trans_indent_modified_file.append(file_path)
                                        globals.list_trans_indent_location_idxs.append(trans_indent_location)
                                        globals.list_trans_indent_level.append(trans_indent_level)
                                    else:                            
                                        idx = globals.list_trans_indent_modified_file.index(file_path)
                                        for i in trans_indent_location:
                                            globals.list_trans_indent_location_idxs[idx].append(i)
                                        for i in trans_indent_level:
                                            globals.list_trans_indent_level[idx].append(i)

                                if "insert_above_inference_line" in loc:
                                    idx_offset = 0
                                elif "insert_below_inference_line" in loc:
                                    if ")" in line:  # e.g. model = Net(xxx)
                                        idx_offset = 1
                                    else:  # e.g. model = Net(xxx, \n xxx, \n xxx)
                                        do_search = True
                                        i_search = 1
                                        while do_search:
                                            following_line = lines[line_idx + i_search]
                                            if ")" in following_line:
                                                do_search = False
                                            i_search += 1
                                            inference_line = \
                                                inference_line + "\n" + \
                                                " " * (get_line_indent_level(line) + 4) + following_line
                                        idx_offset = i_search
                                
                                if "insert_above_inference_line" in loc or "insert_below_inference_line" in loc:
                                    bk_trans_content_this = bk_trans_content[bk_trans_location.index(loc)]

                                    trans_insert_location = line_idx + idx_offset

                                    insert_indent_level = inference_line_indent_level

                                    ### content

                                    # lines to insert
                                    lines_to_insert = bk_trans_content_this
                                    # replace [+] indication with empty
                                    lines_to_insert = lines_to_insert.replace(
                                        "[+] ", " " * insert_indent_level)
                                    # add begin indicator
                                    lines_to_insert = " " * insert_indent_level + "# [NeuralCoder] " + \
                                        self.backend + " [Beginning Line] \n" + lines_to_insert
                                    # replace INDICATIONS with real stuff 
                                    # (for now, inference_line related transformations )
                                    # (have nothing to do with input, dataloader etc, )
                                    # (so no need to put replaces here.)
                                    lines_to_insert = lines_to_insert.replace("\n", " # [coder-enabled]\n")
                                    # add end indicator
                                    lines_to_insert += " # [coder-enabled]\n" + \
                                        " " * insert_indent_level + "# [NeuralCoder] " + \
                                        self.backend + " [Ending Line] # [coder-enabled]"

                                    # customized argument
                                    if self.backend == "pytorch_benchmark":
                                        lines_to_insert = lines_to_insert.replace("NUM_BENCHMARK_ITERATION", 
                                                                                    globals.num_benchmark_iteration)
                                        lines_to_insert = lines_to_insert.replace("ACCURACY_MODE", 
                                                                                    str(False))
                                        lines_to_insert = lines_to_insert.replace("INFERENCE_LINE", 
                                                                                    inference_line.strip())

                                    ### register
                                    
                                    if file_path not in globals.list_trans_insert_modified_file:
                                        globals.list_trans_insert_modified_file.append(file_path)
                                        globals.list_trans_insert_location_idxs.append([trans_insert_location])
                                        globals.list_trans_insert_number_insert_lines.append(
                                            [lines_to_insert.count("\n") + 1]
                                        )
                                        globals.list_trans_insert_lines_to_insert.append([lines_to_insert])
                                    else:
                                        idx = globals.list_trans_insert_modified_file.index(file_path)
                                        globals.list_trans_insert_location_idxs[idx].append(trans_insert_location)
                                        globals.list_trans_insert_number_insert_lines[idx].append(
                                            lines_to_insert.count("\n") + 1
                                        )
                                        globals.list_trans_insert_lines_to_insert[idx].append(lines_to_insert)

                                break # already transformed this line, so skip any further model_name search
                        line_idx += 1

            # PART 3 - for customized location


        logger.debug(
            f"globals.list_trans_insert_modified_file: {globals.list_trans_insert_modified_file}")
        logger.debug(
            f"globals.list_trans_insert_location_idxs: {globals.list_trans_insert_location_idxs}")
        logger.debug(
            f"globals.list_trans_insert_number_insert_lines: {globals.list_trans_insert_number_insert_lines}")
        logger.debug(
            f"globals.list_trans_insert_lines_to_insert: {globals.list_trans_insert_lines_to_insert}")
