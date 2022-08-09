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


from typing import List
from ..utils.line_operation import get_line_indent_level
from .. import globals
import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


# register all relationships of ( [function name] : [return_item] ) pair of the list of code path provided
# but only for "return xxx()" (return a function w/o class prefix) or "return xxx" (return an instance)
# e.g.
# def a1():
#     return b1()
# def b1():
#     return x
# def c():
#     return T.q()
# INPUT: ["example.py"] (above code snippet)
# OUTPUT:
# globals.list_all_function_return_item = ["b1", "x"]
# globals.list_all_function_name = ["a1", "b1"]
def register_func_wrap_pair():
    logger.info(
        f"Analyzing function wrapping relationship for call graph analysis...")
    for path in globals.list_code_path:
        code = open(path, 'r').read()
        lines = code.split('\n')
        line_idx = 0
        is_in_function = False
        func_end_line_idx = -1
        function_def_line_idx = -1
        for line in lines:
            indent_level = get_line_indent_level(line)

            # handle function's end line
            if is_in_function and line_idx == func_end_line_idx:
                is_in_function = False

            # handle function's defnition line, to initiate a function
            if not is_in_function and "def " in line:  # only deal with outermost def
                function_name = line[line.find("def")+4:line.find("(")]

                def_indent_level = get_line_indent_level(line)
                function_def_line_idx = line_idx

                is_in_function = True

                # search for function end line
                search_idx = line_idx + 1
                search_following_lines = True
                multi_comment_flag = False
                while search_following_lines:
                    try:
                        following_line = lines[search_idx]
                    except:  # end of file
                        func_end_line_idx = search_idx
                        break
                    following_indent_level = get_line_indent_level(
                        following_line)

                    # judge_1: indent is equal to def indent
                    judge_1 = following_indent_level <= def_indent_level
                    # judge_2: not starting with")"
                    judge_2 = True if (
                        following_line != "" and following_line[following_indent_level] != ")") else False
                    # judge_3: is not a comment
                    c1 = False
                    c2 = False
                    if multi_comment_flag:
                        c1 = True  # multi-line comment
                    if len(line) > 0 and len(line.lstrip()) > 0 and line.lstrip()[0] == "#":
                        c2 = True  # single-line comment
                    if '"""' in following_line:
                        multi_comment_flag = not multi_comment_flag

                    judge_3 = True if (not c1 and not c2) else False

                    if judge_1 and judge_2 and judge_3:
                        search_following_lines = False
                        func_end_line_idx = search_idx

                    search_idx += 1

                line_idx += 1
                continue

            # handle inside a function
            if is_in_function and line_idx < func_end_line_idx:
                # handle return
                if "return" in line:
                    line_s = line[line.find("return")+7:].strip()
                    # line_s common case: 1. "" 2. "xxx" 3. "xxx, xxx" 3. "xxx()" 4. "xxx(xxx)" 5. "xxx(xxx, xxx)"
                    if line_s == "":  # case 1
                        pass
                    elif line.strip()[0:6] != "return":
                        pass
                    elif 'f"' in line or "#" in line or "if" in line or "." in line or '""' in line or "+" in line:
                        pass
                    elif "(" in line_s:  # case 4 or case 5
                        return_item = line_s[:line_s.find("(")]
                        globals.list_all_function_return_item.append(
                            return_item)
                        globals.list_all_function_name.append(function_name)
                    elif ", " in line_s:  # case 3
                        ls = line_s.split(", ")
                        for return_item in ls:
                            globals.list_all_function_return_item.append(
                                return_item)
                            globals.list_all_function_name.append(
                                function_name)
                    else:  # case 2
                        return_item = line_s
                        globals.list_all_function_return_item.append(
                            return_item)
                        globals.list_all_function_name.append(function_name)

            line_idx += 1
            continue

    logger.debug(
        f"globals.list_all_function_name: {globals.list_all_function_name}")
    logger.debug(
        f"globals.list_all_function_return_item: {globals.list_all_function_return_item}")

# get all wrapper children names of the base function name
# e.g.
# class Net(nn.Module):
#    xxx
#
# def _resnet():
#     model = Net()
#     return model
#
# def resnet34():
#     xxx
#     return _resnet()
#
# def resnet18():
#     xxx
#     return _resnet()
#
# def resnet18_large():
#     xxx
#     return resnet18()
#
# INPUT: "_resnet"
# OUTPUT: ["resnet18", "resnet34", "resnet18_large"]


def get_all_wrap_children(base_function_name: str) -> List:
    length = range(len(globals.list_all_function_return_item))
    base_function_name = [base_function_name]
    do_search = True
    list_child_all = []

    while do_search:
        current_count = len(list_child_all)
        for this_base in base_function_name:
            this_list_child = []
            for i in length:
                if globals.list_all_function_return_item[i] == this_base:
                    this_list_child.append(globals.list_all_function_name[i])
            base_function_name = this_list_child

            list_child_all += this_list_child
            list_child_all = list(set(list_child_all))

        if len(list_child_all) == current_count:
            do_search = False

    return list_child_all
