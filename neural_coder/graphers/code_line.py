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
from .. import globals
from ..utils.line_operation import get_line_indent_level
from ..utils.line_operation import multi_line_comment_detection
from ..utils.line_operation import single_line_comment_or_empty_line_detection
import pprint


class CodeLine:
    def __init__(self):
        self.file_path = None
        self.line_idx = None
        self.line_content = None
        self.indent_level = None
        self.is_multi_line_comment = None
        self.is_single_line_comment_or_empty = None
        self.is_class_def_line = None
        self.is_in_class = None
        self.class_name = None
        self.parent_class_name = None
        self.class_def_line_idx = None
        self.class_end_line_idx = None
        self.is_func_def_line = None
        self.is_in_func = None
        self.func_name = None
        self.func_return_idx = None
        self.return_item = None
        self.func_def_line_idx = None
        self.func_end_line_idx = None

    def print_info(self):
        pp = pprint.PrettyPrinter()
        pp.pprint(self.__dict__)


def register_code_line():
    print_class_related_info = False
    print_func_related_info = False
    print("{:<100} {:<10} {:<20} {:<20} {:<20} {:<40} {:<20} {:<20}".format('line',
                                                                            'line_idx',
                                                                            'is_class_def_line',
                                                                            'is_in_class',
                                                                            'class_name',
                                                                            'parent_class_name',
                                                                            'class_def_line_idx',
                                                                            'class_end_line_idx')) \
        if print_class_related_info else 0
    print("{:<100} {:<10} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format('line',
                                                                                   'line_idx',
                                                                                   'is_func_def_line',
                                                                                   'is_in_func',
                                                                                   'func_name',
                                                                                   'func_return_idx',
                                                                                   'return_item',
                                                                                   'func_def_line_idx',
                                                                                   'func_end_line_idx')) \
        if print_func_related_info else 0
    for path in globals.list_code_path:
        code = open(path, 'r').read()
        lines = code.split('\n')

        line_idx = 0
        is_multi_line_comment = False
        end_multi_line_comment_flag = False

        is_class_def_line = False
        is_in_class = False
        class_name = ""
        parent_class_name = []
        class_def_line_idx = -1
        class_end_line_idx = -1

        is_func_def_line = False
        is_in_func = False
        func_name = ""
        func_return_idx = -1
        return_item = ""
        func_def_line_idx = -1
        func_end_line_idx = -1

        for line in lines:
            CL = CodeLine()
            CL.file_path = path
            CL.line_idx = line_idx
            CL.line_content = line
            CL.indent_level = get_line_indent_level(line)

            is_multi_line_comment, end_multi_line_comment_flag = multi_line_comment_detection(
                line, is_multi_line_comment, end_multi_line_comment_flag)
            CL.is_multi_line_comment = is_multi_line_comment

            is_single_line_comment_or_empty = single_line_comment_or_empty_line_detection(
                line)
            CL.is_single_line_comment_or_empty = is_single_line_comment_or_empty

            # class
            is_class_def_line = False
            if "class " in line and line.lstrip()[0:5] == "class":
                is_in_class = True
                is_class_def_line = True
                line_ls = line.lstrip()
                if "(" in line_ls:  # "class A(B):"
                    class_name = line_ls[line_ls.find(" ")+1:line_ls.find("(")]
                    parent_content = line_ls[line_ls.find(
                        "(")+1:line_ls.find(")")]
                    if "," in parent_content:  # "class A(B, C):"
                        parent_class_name = []
                        parent_content_items = parent_content.split(", ")
                        for parent_content_item in parent_content_items:
                            parent_class_name.append(parent_content_item)
                    else:  # "class A(B):"
                        parent_class_name = [parent_content]
                else:  # "class A:"
                    class_name = line_ls[line_ls.find(" ")+1:line_ls.find(":")]
                    parent_class_name = []

                # search for class end line
                class_def_indent_level = get_line_indent_level(line)
                class_def_line_idx = line_idx
                search_idx = line_idx + 1
                search_following_lines = True
                _is_multi_line_comment = False
                _end_multi_line_comment_flag = False
                while search_following_lines:
                    try:
                        following_line = lines[search_idx]
                    except:  # end of file situation
                        class_end_line_idx = search_idx
                        break
                    following_indent_level = get_line_indent_level(
                        following_line)

                    _is_multi_line_comment, _end_multi_line_comment_flag = multi_line_comment_detection(
                        following_line, _is_multi_line_comment, _end_multi_line_comment_flag)
                    _is_single_line_comment_or_empty = single_line_comment_or_empty_line_detection(
                        following_line)

                    # judge_1: indent is equal to def indent
                    judge_1 = following_indent_level <= class_def_indent_level
                    # judge_2: not starting with")"
                    judge_2 = True if (
                        following_line != "" and following_line[following_indent_level] != ")") else False
                    # judge_3: is not a comment or empty line
                    judge_3 = True if (
                        not _is_multi_line_comment and not _is_single_line_comment_or_empty) else False

                    if judge_1 and judge_2 and judge_3:
                        search_following_lines = False
                        class_end_line_idx = search_idx

                    search_idx += 1

            if is_in_class and line_idx == class_end_line_idx:
                is_in_class = False
                class_name = ""
                parent_class_name = []
                class_def_line_idx = -1
                class_end_line_idx = -1

            # function
            if is_in_func and line_idx == func_end_line_idx:
                is_in_func = False
                func_return_idx = -1
                return_item = ""
                func_name = ""
                func_def_line_idx = -1
                func_end_line_idx = -1

            is_func_def_line = False
            # only consider outermost function, not consider def(def())
            if not is_in_func and "def " in line:
                is_in_func = True
                is_func_def_line = True
                func_name = line[line.find("def")+4:line.find("(")]

                # search for func end line
                func_def_indent_level = get_line_indent_level(line)
                func_def_line_idx = line_idx
                search_idx = line_idx + 1
                search_following_lines = True
                _is_multi_line_comment = False
                _end_multi_line_comment_flag = False
                while search_following_lines:
                    try:
                        following_line = lines[search_idx]
                    except:  # end of file situation
                        func_end_line_idx = search_idx
                        break
                    following_indent_level = get_line_indent_level(
                        following_line)

                    if "return" in following_line:
                        func_return_idx = search_idx
                        return_item = following_line[following_line.find(
                            "return")+7:].strip()

                    _is_multi_line_comment, _end_multi_line_comment_flag = multi_line_comment_detection(
                        following_line, _is_multi_line_comment, _end_multi_line_comment_flag)
                    _is_single_line_comment_or_empty = single_line_comment_or_empty_line_detection(
                        following_line)

                    # judge_1: indent is equal to def indent
                    judge_1 = following_indent_level <= func_def_indent_level
                    # judge_2: not starting with")"
                    judge_2 = True if (
                        following_line != "" and following_line[following_indent_level] != ")") else False
                    # judge_3: is not a comment or empty line
                    judge_3 = True if (
                        not _is_multi_line_comment and not _is_single_line_comment_or_empty) else False

                    if judge_1 and judge_2 and judge_3:
                        search_following_lines = False
                        func_end_line_idx = search_idx

                    search_idx += 1

            CL.is_class_def_line = is_class_def_line
            CL.is_in_class = is_in_class
            CL.class_name = class_name
            CL.parent_class_name = parent_class_name
            CL.class_def_line_idx = class_def_line_idx
            CL.class_end_line_idx = class_end_line_idx
            print("{:<100} {:<10} {:<20} {:<20} {:<20} {:<40} {:<20} {:<20}".format(line[0:100],
                                                                                    line_idx,
                                                                                    is_class_def_line,
                                                                                    is_in_class,
                                                                                    class_name,
                                                                                    str(
                parent_class_name),
                class_def_line_idx,
                class_end_line_idx)) if print_class_related_info else 0

            CL.is_func_def_line = is_func_def_line
            CL.is_in_func = is_in_func
            CL.func_name = func_name
            CL.func_return_idx = func_return_idx
            CL.return_item = return_item
            CL.func_def_line_idx = func_def_line_idx
            CL.func_end_line_idx = func_end_line_idx
            print("{:<100} {:<10} {:<20} {:<20} \
                {:<20} {:<20} {:<20} {:<20} {:<20}".format(line[0:100],
                                                           line_idx,
                                                           is_func_def_line,
                                                           is_in_func,
                                                           func_name,
                                                           func_return_idx,
                                                           return_item[0:20],
                                                           func_def_line_idx,
                                                           func_end_line_idx)) \
                if print_func_related_info else 0

            globals.list_code_line_instance.append(CL)
            line_idx += 1

    # for i in globals.list_code_line_instance:
    #     i.print_info()
