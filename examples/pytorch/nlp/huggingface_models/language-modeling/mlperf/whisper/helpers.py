# Copyright 2025 Intel Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from typing import List
from legacy_helpers import __levenshtein

def expand_concatenations(words_list: List, reference_dict: dict, reference_list: List):
    """
    Finds matching compound words in 'words_list' which exist as keys in 'reference_dict', if any.
    If found, the compound word will be separated using reference_dict if the substitution reduces
    the 'Levenshtein distance' between 'words_list' and 'reference_list'.
    Args:
        words_list: List of English word strings
        reference_dict: Dictionary mapping compound words to a list a separated word strings.
        reference_list: List of English word strings
    Returns:
        Modified 'word_string' with compound words replaced by individual strings, if any
    """
    score = __levenshtein(words_list, reference_list)

    # Searches each word in 'word_list' for separability using the reference list. Once all options are
    # considered, the modified 'word_list' is returned. Length of 'word_list' can grow, but not contract.
    i = 0
    words_length = len(words_list)
    while i < words_length:
        if words_list[i] in reference_dict.keys():
            words_candidate = words_list[:i] + reference_dict[words_list[i]] + words_list[i + 1:]

            # If levenshtein distance reduced, cache new word_list and resume search
            candidate_levenshtein = __levenshtein(words_candidate, reference_list)
            if candidate_levenshtein < score:
                words_list = words_candidate
                words_length = len(words_list)
                score = candidate_levenshtein
        i += 1
    return words_list

def get_expanded_wordlist(words_list: List, reference_list: List):
    """
    Provided two lists of English words, the two will be compared, and any compound words found in
        'word_list' which are separated in 'reference_list' will be separated and the modified
        'word_list' will be returned.
    Args:
        word_list: List of English word strings
        reference_list: List of English word strings
    Returns:
        List of words modified from 'word_list' after expanding referenced compound words
    """

    # If levenshtein distance < 2, there cannot be any compound word separation issues.
    if __levenshtein(words_list, reference_list) < 2:
        return words_list

    # Adding two-word compouding candidates to checklist
    checklist = {}
    for i in range(len(reference_list) - 1):
        compound = reference_list[i] + reference_list[i + 1]
        checklist[compound] = [reference_list[i], reference_list[i + 1]]

    # Adding three-word compounding candidates to checklist
    for i in range(len(reference_list) - 2):
        compound = reference_list[i] + reference_list[i + 1] + reference_list[i + 2]
        checklist[compound] = [reference_list[i], reference_list[i + 1], reference_list[i + 2]]

    # All compiled candidates will be checked, and after checking for minimal Levenshtein
    # distance, the modified list (or original if compounding not found) is directly returned 
    return expand_concatenations(words_list, checklist, reference_list)
