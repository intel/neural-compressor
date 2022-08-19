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
import logging

logging.basicConfig(level=globals.logging_level,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S +0000')
logger = logging.getLogger(__name__)


def execute_insert_transformation(list_transformed_code):
    """Insert code lines into file."""
    for index, file_path in enumerate(globals.list_trans_insert_modified_file):
        trans_location_idxs = globals.list_trans_insert_location_idxs[index]
        trans_number_insert_lines = globals.list_trans_insert_number_insert_lines[index]
        trans_lines_to_insert = globals.list_trans_insert_lines_to_insert[index]

        # sort trans_location_idxs and sort the other lists accordingly
        trans_number_insert_lines = [
            i for _, i in sorted(zip(trans_location_idxs, trans_number_insert_lines))
        ]
        trans_lines_to_insert = [
            i for _, i in sorted(zip(trans_location_idxs, trans_lines_to_insert))
        ]
        trans_location_idxs = sorted(trans_location_idxs)
        
        file_path_idx = globals.list_code_path.index(file_path)
        lines_transformed = list_transformed_code[file_path_idx].split('\n')

        # math
        t = [0]
        u = 0
        for n in trans_number_insert_lines:
            u = u + n
            t.append(u)
        t = t[:-1]
        
        logger.debug(f"t: {t}")
        trans_location_idxs = [sum(i) for i in zip(trans_location_idxs, t)]
        logger.debug(f"trans_location_idxs after adjustment: {trans_location_idxs}")

        for idx in trans_location_idxs:  # actual transformation (insertion)
            additions = trans_lines_to_insert[trans_location_idxs.index(idx)].split("\n")
            additions = additions[::-1]  # reverse
            for i in range(len(additions)):
                lines_transformed.insert(idx, additions[i])

        # transfer lines_transformed to code format ("\n" save write)
        code_transformed = "".join([i + "\n" for i in lines_transformed])[0:-1]

        list_transformed_code[file_path_idx] = code_transformed

    return list_transformed_code


def execute_indent_transformation(list_transformed_code):
    """Indent code lines with spaces at the beginning."""
    for index, file_path in enumerate(globals.list_trans_indent_modified_file):
        trans_location_idxs = globals.list_trans_indent_location_idxs[index]
        trans_indent_level = globals.list_trans_indent_level[index]

        file_path_idx = globals.list_code_path.index(file_path)
        lines_transformed = list_transformed_code[file_path_idx].split('\n')

        for idx in trans_location_idxs:  # actual transformation (indent)
            this_indent_level = trans_indent_level[trans_location_idxs.index(idx)]
            lines_transformed[idx] = " " * 4 * this_indent_level + lines_transformed[idx]

        # transfer lines_transformed to code format ("\n" save write)
        code_transformed = "".join([i + "\n" for i in lines_transformed])[0:-1]

        list_transformed_code[file_path_idx] = code_transformed

    return list_transformed_code
