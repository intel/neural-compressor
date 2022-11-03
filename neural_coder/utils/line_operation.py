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

# get line's indent level
def get_line_indent_level(line: str) -> int:
    if list(set(line)) == [" "]:
        return 0
    else:
        return len(line) - len(line.lstrip())


# determine if line is multi-line comment
def multi_line_comment_detection(line: str, previous_line_is_multi_line_comment: bool, end_ml_comment_flag: bool):
    this_line_is_multi_line_comment = previous_line_is_multi_line_comment

    if previous_line_is_multi_line_comment:
        if end_ml_comment_flag:
            this_line_is_multi_line_comment = False
        else:
            this_line_is_multi_line_comment = True
        if '"""' in line:  # end of multi line comment
            end_ml_comment_flag = True
        else:
            end_ml_comment_flag = False
    else:
        if '"""' in line and line.count('"') == 3:  # start of multi line comment, e.g. [    """]
            this_line_is_multi_line_comment = True
            end_ml_comment_flag = False
        else:
            this_line_is_multi_line_comment = False
            end_ml_comment_flag = False

    if not this_line_is_multi_line_comment:
        if '"""' in line:
            this_line_is_multi_line_comment = True
            end_ml_comment_flag = False

    if '"""' in line and line.count('"') == 6:
        this_line_is_multi_line_comment = True
        end_ml_comment_flag = True

    return this_line_is_multi_line_comment, end_ml_comment_flag


# determine if line is single-line comment or empty
def single_line_comment_or_empty_line_detection(line: str) -> bool:
    this_line_is_single_line_comment_or_empty_line = False

    if len(line) == 0 or line.isspace():  # empty line or all spaces
        this_line_is_single_line_comment_or_empty_line = True
    elif '"""' in line and line.count('"') == 6:  # e.g. [    """ some single-line comment """]
        this_line_is_single_line_comment_or_empty_line = True
    # e.g. [    # some single-line comment]
    elif len(line) > 0 and len(line.lstrip()) > 0 and line.lstrip()[0] == "#":
        this_line_is_single_line_comment_or_empty_line = True

    return this_line_is_single_line_comment_or_empty_line


# determine if line is a eval func of model_name, 
# like "xxx = model_name(yyy)" or "model_name(yyy)" or "model_name.some_func(yyy)"
def is_eval_func_model_name(model_name: str, line: str) -> str:
    line_ = line.replace(' ', '')
    # model(input)
    judge_1 = line_.find(model_name + "(") > -1
    judge_2 = (line_.find("=") > 0 and
               line_.find("=") < line_.find(model_name) and
               line_[line_.find("=")+1:line_.find("(")] == model_name) or line_.find(model_name) == 0
    # model.some_func(input)
    judge_3 = line_.find(model_name + ".") > -1
    judge_4 = line_.find("(") > -1
    judge_5 = (line_.find("=") > 0 and
               line_.find("=") < line_.find(model_name) and
               line_[line_.find("=")+1:line_.find(".")] == model_name) or line_.find(model_name) == 0
    exclude_function_list = [
        "__init__",
        "to",
        "eval",
        "train",
        "optimize",
        "model",
        "framework",
        "config",
        "load_state_dict",
        "cpu",
        "cuda",
        "contiguous",
        "features",
        "freeze_feature_encoder",
    ]
    judge_6 = line_[line_.find(".")+1:line_.find("(")] not in exclude_function_list
    judge_7 = "model.config" not in line and "model.features" not in line
    judge_8 = "model(**inputs)" in line

    if judge_1 and judge_2:
        return True, "forward"
    elif judge_3 and judge_4 and judge_5 and judge_6 and judge_7:
        return True, "non-forward"
    elif judge_8:
        return True, "non-forward"
    else:
        return False, "not an eval func"


# get lhs of line of format "xxx = yyy"
def get_line_left_hand_side(line: str) -> str:
    line_ = line.replace(' ', '')
    lhs = line_[:line_.find("=")]
    return lhs


# determine if line is for format "xxx = yyy(zzz)" and get lhs and rhs of "="
def of_definition_format(line: str):
    line_ = line.replace(' ', '')
    is_def = False
    lhs = ""
    rhs = ""
    if "=" in line_ and "(" in line_ and line_.find("=") < line_.find("("):
        is_def = True
        lhs = line_[:line_.find("=")]
        rhs = line_[line_.find("=")+1:line_.find("(")]
        if "." not in rhs:
            pass
        else:
            rhs = rhs[rhs.find(".")+1:]
    return is_def, lhs, rhs


# get the line without comment
def get_line_wo_comment(line: str):
    line = line[:line.find("#")].rstrip()
    return line
