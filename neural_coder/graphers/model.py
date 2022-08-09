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


# FOR PYTORCH ONLY

from .function import get_all_wrap_children
import pprint
import re
from ..utils.line_operation import get_line_indent_level, of_definition_format
from typing import List
from .. import globals
import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


class ClassDefinition:
    # (class_name + file_path) is the unique determination
    def __init__(self, class_name, file_path, class_def_line_idx, parent_class_name):
        self.class_name = class_name
        self.file_path = file_path
        self.class_def_line_idx = class_def_line_idx
        self.parent_class_name = parent_class_name

    def print_info(self):
        logger.debug(f"ClassDefinitionprint_info(): {self.__dict__}")


class ModelDefinition:
    def __init__(self,
                 model_name,
                 class_name,
                 file_path,
                 model_def_line_idx,
                 function_def_line_idx,
                 function_name
                 ):
        self.model_name = model_name
        self.class_name = class_name
        self.file_path = file_path
        self.model_def_line_idx = model_def_line_idx
        self.function_def_line_idx = function_def_line_idx
        self.function_name = function_name

    def print_info(self):
        logger.debug(f"ModelDefinition.print_info(): {self.__dict__}")


# search nnModule classes
def register_nnModule_class():
    logger.info(f"Analyzing nn.Module class definitions in all files ...")
    # search raw nnModule class (e.g. class ClassName(nn.Module):)
    for cl in globals.list_code_line_instance:
        parent_class_has_nnModule = list(set(cl.parent_class_name) & set(
            ["nn.Module", "torch.nn.Module", "nn.Sequential", "torch.Sequential", "_BaseAutoModelClass"])) != []
        if cl.is_class_def_line and parent_class_has_nnModule:
            CD = ClassDefinition(class_name=cl.class_name,
                                 file_path=cl.file_path,
                                 class_def_line_idx=cl.class_def_line_idx,
                                 parent_class_name=cl.parent_class_name)
            CD.print_info()
            globals.list_class_name.append(cl.class_name)
            globals.list_class_def_instance.append(CD)

    # search child class of nnModule class (recursively),  (e.g. class ClassName(ClassNameFather):)
    # this is to complete the nnModule class list
    # e.g. A(nn.Module), B(A), C(B), D(C)
    search_scope = globals.list_class_name
    do_search = True
    while do_search:
        list_child_class_name = []
        for cl in globals.list_code_line_instance:
            parent_class_has_nnModule = list(
                set(cl.parent_class_name) & set(search_scope)) != []
            if cl.is_class_def_line and parent_class_has_nnModule:
                CD = ClassDefinition(class_name=cl.class_name,
                                     file_path=cl.file_path,
                                     class_def_line_idx=cl.class_def_line_idx,
                                     parent_class_name=cl.parent_class_name)
                CD.print_info()
                globals.list_class_name.append(cl.class_name)
                globals.list_class_def_instance.append(CD)
                list_child_class_name.append(cl.class_name)
        search_scope = list_child_class_name
        if len(search_scope) == 0:
            do_search = False

    # unique
    globals.list_class_name = list(set(globals.list_class_name))

    logger.debug(f"class name count: {len(globals.list_class_name)}")
    logger.debug(f"class name list : {globals.list_class_name}")


# search nnModule instance definition
def register_nnModule_instance_definition():
    logger.info(
        f"Analyzing nn.Module instance (model instance) definitions in all files ...")
    # search model definition lines like "model_name = ClassName(xxx)"
    def_cl = []
    for cl in globals.list_code_line_instance:
        if not cl.is_multi_line_comment and not cl.is_single_line_comment_or_empty:
            is_def, lhs, rhs = of_definition_format(cl.line_content)
            if is_def and \
               rhs in globals.list_class_name + ["Module", "Sequential"] and \
               cl.class_name not in globals.list_class_name and \
               "(" not in cl.return_item:
                def_cl.append(cl)

    list_lhs = []
    list_rhs = []
    list_is_in_func = []
    list_func_name = []
    list_return_item = []
    list_file_path = []
    list_line_idx = []
    list_func_def_line_idx = []
    for cl in def_cl:
        is_def, lhs, rhs = of_definition_format(cl.line_content)
        list_lhs.append(lhs)
        list_rhs.append(rhs)
        list_is_in_func.append(cl.is_in_func)
        list_func_name.append(cl.func_name)
        list_return_item.append(cl.return_item)
        list_file_path.append(cl.file_path)
        list_line_idx.append(cl.line_idx)
        list_func_def_line_idx.append(cl.func_def_line_idx)

    # register qualified model's name of lines like "model_name = ClassName(xxx)"
    globals.list_wrapper_base_function_name = []
    for i in range(len(list_lhs)):
        # situation 1: "model = Net()" outside any function
        if not list_is_in_func[i] and "tokenizer" not in list_lhs[i]:
            # register this model
            globals.list_model_name.append(list_lhs[i])
            MD = ModelDefinition(model_name=list_lhs[i],
                                 class_name=list_rhs[i],
                                 file_path=list_file_path[i],
                                 model_def_line_idx=list_line_idx[i],
                                 function_def_line_idx=-1,
                                 function_name="null")  # this MD is for all models defined outside a function
            MD.print_info()
            globals.list_model_def_instance.append(MD)
        elif list_is_in_func[i]:  # situation 2: "model = Net()" is inside a function
            # situation 2-1: the function does not return another model's name, and is not __init__
            if list_return_item[i] not in list_lhs and \
                    list_func_name[i] != "__init__" and "tokenizer" not in list_lhs[i]:
                # register this model
                globals.list_model_name.append(list_lhs[i])
                MD = ModelDefinition(model_name=list_lhs[i],
                                     class_name=list_rhs[i],
                                     file_path=list_file_path[i],
                                     model_def_line_idx=list_line_idx[i],
                                     function_def_line_idx=list_func_def_line_idx[i],
                                     function_name=list_func_name[i])  \
                    # this MD is for all models defined outside a function
                MD.print_info()
                globals.list_model_def_instance.append(MD)
            # situation 2-2: the function returns another model's name
            elif list_return_item[i] in list_lhs:
                globals.list_wrapper_base_function_name.append(
                    list_func_name[i])

    # register function_name like "xxx" in "def xxx() ... return NNModuleClass()"
    for cl in globals.list_code_line_instance:
        if cl.is_in_func and cl.line_idx == cl.func_return_idx \
                and cl.return_item[:cl.return_item.find("(")] in globals.list_class_name:
            globals.list_wrapper_base_function_name.append(cl.func_name)

    # for all base function_name (that returns nnModule instance),
    # find all wrapper function_name of the base wrapper function_name
    globals.list_wrapper_base_function_name = list(
        set(globals.list_wrapper_base_function_name))
    globals.list_wrapper_children_function_name = []
    for i in globals.list_wrapper_base_function_name:
        globals.list_wrapper_children_function_name += get_all_wrap_children(i)
    globals.list_wrapper_all_function_name = globals.list_wrapper_base_function_name + \
        globals.list_wrapper_children_function_name
    globals.list_wrapper_all_function_name = list(
        set(globals.list_wrapper_all_function_name))

    # register function_name like "xxx" in "def xxx() ... model = some_wrapper_function() ... return model"
    for cl in globals.list_code_line_instance:
        if cl.is_in_func and not cl.is_multi_line_comment and not cl.is_single_line_comment_or_empty:
            is_def, lhs, rhs = of_definition_format(cl.line_content)
            if is_def and \
               rhs in globals.list_wrapper_all_function_name and \
               cl.class_name not in globals.list_class_name and \
               cl.return_item == lhs:
                globals.list_wrapper_base_function_name.append(cl.func_name)

    # (again)
    # for all base function_name (that returns nnModule instance),
    # find all wrapper function_name of the base wrapper function_name
    globals.list_wrapper_base_function_name = list(
        set(globals.list_wrapper_base_function_name))
    for i in globals.list_wrapper_base_function_name:
        globals.list_wrapper_children_function_name += get_all_wrap_children(i)
    globals.list_wrapper_all_function_name += globals.list_wrapper_base_function_name + \
        globals.list_wrapper_children_function_name
    globals.list_wrapper_all_function_name = list(
        set(globals.list_wrapper_all_function_name))

    # print all wrapper function names for debug purpose
    logger.debug(
        f"globals.list_wrapper_all_function_name: {globals.list_wrapper_all_function_name}")

    for cl in globals.list_code_line_instance:
        if not cl.is_multi_line_comment and not cl.is_single_line_comment_or_empty and cl.func_name != "__init__":
            is_def, lhs, rhs = of_definition_format(cl.line_content)
            if is_def and \
                rhs in globals.list_wrapper_all_function_name and \
                rhs not in ["self.model", "model", "self.call", "call"] and \
                "forward" not in rhs and \
                "config" not in lhs and "congfig" not in lhs and "," not in lhs and \
                "inference" not in lhs and "tokenizer" not in lhs and \
                cl.class_name not in globals.list_class_name and \
                    cl.func_name not in globals.list_wrapper_all_function_name:
                # register this model
                globals.list_model_name.append(lhs)
                MD = ModelDefinition(model_name=lhs,
                                     class_name="(note: this is a func-defined model)"+rhs,
                                     file_path=cl.file_path,
                                     model_def_line_idx=cl.line_idx,
                                     function_def_line_idx=cl.func_def_line_idx,
                                     function_name=cl.func_name)
                # this MD is for all models defined by a wrapper function (e.g. model = models.resnet18())
                # model def can be outside any function, or in a function that is not itself a wrapper function
                MD.print_info()
                globals.list_model_def_instance.append(MD)

    globals.list_model_name = list(set(globals.list_model_name))

    # print all model names for debug purpose
    logger.debug(f"model name list: {globals.list_model_name}")
