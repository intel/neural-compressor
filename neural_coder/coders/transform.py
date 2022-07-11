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


from .. import globals

# [insert] some code lines into file


def execute_insert_transformation(list_transformed_code):
    for file_path in globals.list_trans_insert_modified_file:
        trans_location_idxs = globals.list_trans_insert_location_idxs[globals.list_trans_insert_modified_file.index(
            file_path)]
        trans_number_insert_lines = \
            globals.list_trans_insert_number_insert_lines[globals.list_trans_insert_modified_file.index(
                file_path)]
        trans_lines_to_insert = \
            globals.list_trans_insert_lines_to_insert[globals.list_trans_insert_modified_file.index(
                file_path)]

        file_path_idx = globals.list_code_path.index(file_path)
        lines_transformed = list_transformed_code[file_path_idx].split('\n')

        # this part is for "insert" kind of transformation only (math)
        t = [0]
        u = 0
        for n in trans_number_insert_lines:
            u = u + n
            t.append(u)
        t = t[:-1]

        trans_location_idxs = [sum(i) for i in zip(trans_location_idxs, t)]

        for idx in trans_location_idxs:  # actual transformation (insertion)
            additions = trans_lines_to_insert[trans_location_idxs.index(
                idx)].split("\n")
            additions = additions[::-1]  # reverse
            for i in range(len(additions)):
                lines_transformed.insert(idx, additions[i])

        # transfer lines_transformed to code format ("\n" save write)
        code_transformed = "".join([i + "\n" for i in lines_transformed])[0:-1]

        list_transformed_code[file_path_idx] = code_transformed

    return list_transformed_code

# [indenting] some code lines with " " into file


def execute_indenting_transformation(list_transformed_code):
    for file_path in globals.list_trans_indenting_modified_file:
        trans_location_idxs = \
            globals.list_trans_indenting_location_idxs[globals.list_trans_indenting_modified_file.index(
                file_path)]
        trans_indenting_level = \
            globals.list_trans_indenting_level[globals.list_trans_indenting_modified_file.index(
                file_path)]

        file_path_idx = globals.list_code_path.index(file_path)
        lines_transformed = list_transformed_code[file_path_idx].split('\n')

        for idx in trans_location_idxs:  # actual transformation (indenting)
            this_indenting_level = trans_indenting_level[trans_location_idxs.index(
                idx)]
            lines_transformed[idx] = " " * 4 * \
                this_indenting_level + lines_transformed[idx]

        # transfer lines_transformed to code format ("\n" save write)
        code_transformed = "".join([i + "\n" for i in lines_transformed])[0:-1]

        list_transformed_code[file_path_idx] = code_transformed

    return list_transformed_code
