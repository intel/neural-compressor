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

def fp8_matmul_swap(list_transformed_code, list_file_path):

    count = 0
    for cl in globals.list_code_line_instance:
        if " = torch.matmul" in cl.line_content or " = torch.bmm" in cl.line_content:
            # print(cl.line_content)
            # # find the file of this cl (where to edit the lines)
            # print(cl.file_path)
            # # find the Class name of this cl
            # print(cl.class_name)
            # # find __init__ func location of this Class
            # print(cl.class_def_line_idx)

            file_path = cl.file_path
            class_def_line_idx = cl.class_def_line_idx
            class_name = cl.class_name
            line_content = cl.line_content
            line_idx = cl.line_idx

            if "modeling_" not in file_path:
                continue

            if class_def_line_idx != -1:
                # get file content from list_transformed_code
                for idx, path in enumerate(list_file_path):
                    if path == file_path:
                        path_idx = idx
                        break
                file = list_transformed_code[path_idx]
                lines = file.split('\n')
                
                # determine if have __init__(), only have_init conduct code changes
                have_init = False
                for i in range(class_def_line_idx, len(lines)): # from class_def line to end of file
                    if "def " in lines[i] and "__init__" not in lines[i]: # does not have init
                        have_init = False
                        break
                    if "def __init__(" in lines[i]: # have init
                        have_init = True
                        init_line_idx = i
                        super_found = False
                        while not super_found:
                            i += 1
                            if "super().__init__(" in lines[i]:
                                super_found = True
                                init_super_idx = i
                        break
                # print(have_init)
                
                if have_init: # conduct transformation
                    # print("count = ", count)
                    # swap sentences to self-defined torch_matmul torch_bmm
                    if " = torch.matmul" in line_content:
                        new_line_content = line_content.replace(" = torch.matmul", " = self.torch_matmul_" + str(count))
                    elif " = torch.bmm" in line_content:
                        new_line_content = line_content.replace(" = torch.bmm", " = self.torch_bmm_" + str(count))
                    lines[line_idx] = new_line_content
                    
                    # # insert definition to location: init_line_idx + 2
                    # insertion0 = "        from mpemu.module_wrappers import Matmul, BatchMatmul"
                    # if " = torch.matmul" in line_content:
                        # insertion1 = "        self.torch_matmul_" + str(count) + " = Matmul()"
                    # elif " = torch.bmm" in line_content:
                        # insertion1 = "        self.torch_bmm_" + str(count) + " = BatchMatmul()"
                    # lines.insert(init_line_idx + 2, insertion0)
                    # lines.insert(init_line_idx + 2, insertion1)
                    
                    super_line = lines[init_super_idx]
                    if " = torch.matmul" in line_content:
                        new_super_line = super_line + "; from mpemu.module_wrappers import Matmul; self.torch_matmul_" + str(count) + " = Matmul()"
                    elif " = torch.bmm" in line_content:
                        new_super_line = super_line + "; from mpemu.module_wrappers import BatchMatmul; self.torch_bmm_" + str(count) + " = BatchMatmul()"
                    lines[init_super_idx] = new_super_line
                    # print(new_super_line)
                    
                    # put transformed code back into interface.py through list_transformed_code
                    list_transformed_code[path_idx] = '\n'.join(lines)
                    
                    count += 1
                else:
                    pass

    return list_transformed_code


def fp8_add_swap(list_transformed_code, list_file_path):

    count = 0
    for cl in globals.list_code_line_instance:

        to_swap = False
        
        line_content = cl.line_content
        
        case = -1
        
        # case 0: xxx += xxx
        # case 1: xxx = func( xxx + xxx )
        # case 2: xxx = ( xxx ) + xxx
        # case 3: xxx = xxx + ( xxx )
        # case 4: xxx = xxx + xxx
        
        if cl.func_name == "forward": # only change things in forward
            if "+" in line_content \
                and "=" in line_content \
                and line_content.count("+") == 1 \
                and ((line_content.count("(") == 0 and line_content.count(")") == 0) or (line_content.count("(") == 1 and line_content.count(")") == 1)) \
                and ((line_content.find("=") < line_content.find("+")) or ("+=" in line_content)) \
                and "..." not in line_content: 
            # only 1 + and up to 1 () is currently supported, i.e. xxx = xxx + func(xxx + xxx) is not; multi-line is not
                to_swap = True
                if "+=" in line_content: # xxx += xxx
                    case = 0
                elif "(" in line_content and line_content.find("=") < line_content.find("("): # deal with ( ) case
                    if line_content.find("(") < line_content.find("+") and line_content.find(")") > line_content.find("+"): # xxx = func( xxx + xxx )
                        case = 1
                    elif line_content.find(")") < line_content.find("+"): # xxx = ( xxx ) + xxx
                        case = 2
                    elif line_content.find("(") > line_content.find("+"): # xxx = xxx + ( xxx )
                        case = 3
                else: # simple xxx = xxx + xxx
                    case = 4
            else:
                to_swap = False
        else:
            to_swap = False
        
        if to_swap:
            line_content_backup = line_content
            if "#" in line_content: # has comment, remove comment
                line_content = line_content[0:line_content.find("#")]
            if case == 0:
                add_target = line_content[0:line_content.find("+")].strip()
                add_left = line_content[0:line_content.find("+")].strip()
                add_right = line_content[line_content.find("=")+1:].strip()
            elif case == 1:
                add_left = line_content[line_content.find("(")+1:line_content.find("+")].strip()
                add_right = line_content[line_content.find("+")+1:line_content.find(")")].strip()
            elif (case == 2 or case == 3 or case == 4):
                add_target = line_content[0:line_content.find("=")].strip()
                add_left = line_content[line_content.find("=")+1:line_content.find("+")].strip()
                add_right = line_content[line_content.find("+")+1:].strip()
            line_content = line_content_backup

        if to_swap:
            # print(cl.line_content)
            # # find the file of this cl (where to edit the lines)
            # print(cl.file_path)
            # # find the Class name of this cl
            # print(cl.class_name)
            # # find __init__ func location of this Class
            # print(cl.class_def_line_idx)

            file_path = cl.file_path
            class_def_line_idx = cl.class_def_line_idx
            class_name = cl.class_name
            line_idx = cl.line_idx
            indent_level = cl.indent_level

            # if "modeling_" not in file_path:
                # continue

            if class_def_line_idx != -1:
                # get file content from list_transformed_code
                for idx, path in enumerate(list_file_path):
                    if path == file_path:
                        path_idx = idx
                        break
                file = list_transformed_code[path_idx]
                lines = file.split('\n')
                
                # determine if have __init__(), only have_init conduct code changes
                have_init = False
                for i in range(class_def_line_idx, len(lines)): # from class_def line to end of file
                    if "def " in lines[i] and "__init__" not in lines[i]: # does not have init
                        have_init = False
                        break
                    if "def __init__(" in lines[i]: # have init
                        have_init = True
                        init_line_idx = i
                        super_found = False
                        while not super_found:
                            i += 1
                            try:
                                if "super().__init__(" in lines[i]:
                                    super_found = True
                                    init_super_idx = i
                            except:
                                have_init = False
                                break
                        break
                # print(have_init)
                
                if have_init: # conduct transformation
                    # print("count = ", count)
                    # swap sentences to self-defined torch_add
                    if case == 1:
                        replace_content = "self.torch_add_" + str(count) + "(" + add_left + ", " + add_right + ")"
                        new_line_content = line_content[0:line_content.find("(")+1] + replace_content + line_content[line_content.find(")"):]
                    elif (case == 0 or case == 2 or case == 3 or case == 4):
                        new_line_content = indent_level * " " + add_target + " = self.torch_add_" + str(count) + "(" + add_left + ", " + add_right + ")"
                    lines[line_idx] = new_line_content
                    
                    # # insert definition to location: init_line_idx + 2
                    # insertion0 = "        from mpemu.module_wrappers import Matmul, BatchMatmul"
                    # if " = torch.matmul" in line_content:
                        # insertion1 = "        self.torch_matmul_" + str(count) + " = Matmul()"
                    # elif " = torch.bmm" in line_content:
                        # insertion1 = "        self.torch_bmm_" + str(count) + " = BatchMatmul()"
                    # lines.insert(init_line_idx + 2, insertion0)
                    # lines.insert(init_line_idx + 2, insertion1)
                    
                    super_line = lines[init_super_idx]
                    new_super_line = super_line + "; from mpemu.module_wrappers import EltwiseAdd; self.torch_add_" + str(count) + " = EltwiseAdd()"
                    lines[init_super_idx] = new_super_line
                    # print(new_super_line)
                    
                    # put transformed code back into interface.py through list_transformed_code
                    list_transformed_code[path_idx] = '\n'.join(lines)
                    
                    count += 1
                else:
                    pass

    return list_transformed_code
