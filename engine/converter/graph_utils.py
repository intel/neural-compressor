# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

from neural_compressor.utils import logger
import copy
import re
import numpy as np
from collections import namedtuple, OrderedDict
from schema import Schema, And, Or


DTYPES_DICT = {"float16": "fp16",
               "float32": "fp32",
               "int8": "s8",
               "int32": "s32",
               "int64": "s64",
               "uint8": "u8",
               "uint16": "u16",
               }


def names_from_input(name):
    """Static method that get the valid node / tensor name from input name.
    Args:
        name (string): name defined in the input field.
    Returns:
        string tuple: (node's name, tensor's name)

    for example: In NodeDef.input, the name from list is tensor name, may not end with ':0',
                which can not be used for tensor name in the new Graph class. If it end with ':0',
                it can not also be used for node name in the new Graph class
    """
    if name.startswith("^"):
        name = name[1:]
    m = re.search(r"(.*)(:\d+)$", name)
    # not end with ':x"
    if m is None:
        node_name = name
        tensor_name = name + ':0'
    # end with ':x'
    else:
        node_name = m.group(1)
        tensor_name = name

    return (node_name, tensor_name)


def get_data_dtype(data):
    """Get the const data dtype
    Args:
       data (numpy data): a const data to model
    Returns:
       dtype (String): the value in DTYPES_DICT
    """
    dtype = None
    if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating):
        try:
            dtype = DTYPES_DICT[str(data.dtype[0])]
        except BaseException:
            dtype = DTYPES_DICT[str(data.dtype)]

    return dtype


def search_straight_pattern(input_pattern, graph):
    """search user specified patterns on internal grpah structure.
    Attention: the input computation chain in the graph which can be called pattern, there must be 
                straight (or sequence). It means it has not any subgraph nodes. Otherwise this 
                function returns []
    Args:
        input_pattern (list): Contains the op_type of the nodes in pattern. The element of the 
        list could be string/list/tuple, string or list means the specified op_type are mandatory 
        while tuple stands for optional.
        For example, a input pattern mybe like this:
        ['Mul', 'Mul', ['Add', 'AddV2']] it equals to below patterns:
        'Mul' + 'Mul' + 'Add'
        'Mul' + 'Mul' + 'AddV2'
        graph: Graph Class, the new graph generated from extractor.
    Returns: [string list]. The length is the matched pattern results in the graph, for example,
            the graph has 24 layers and each layer has a 'LayerNorm' pattern, then the length is 
            24. Each match pattern result is still a list contains the node names, and the last
            element is the op_type list corresponding to the former node names. 
            For example, the return result maybe like this:
            [
                ['Mul' node name,
                'Mul' node name,
                'Add' node name,
                ['Mul', 'Mul', 'Add']],
                    
                ['Mul' node name,
                'Mul' node name,
                'AddV2' node name,
                ['Mul', 'Mul', 'AddV2']],

                ...
            ]
    """
    def _validate_input(data, creteria):
        if isinstance(creteria, str) and data == creteria:
            return True

        if isinstance(creteria, (list, tuple)) and data in creteria:
            return True

        return False

    def _compare_list(list_a, list_b):
        """Check list a is a subset of list b.
        e.g, list a is ['a', 'b', 'c'] while list b is ['a', 'b', 'c', 'd'],
        then list a is subset of list b.
        Args:
            list_a ([Any]): list A
            list_b ([Any]): list B
        Returns:
            [bool]: list a is a subset of list b or not.
        """
        assert isinstance(list_a, list)
        assert isinstance(list_b, list)
        is_subset = True

        for index, value in enumerate(list_a):
            is_subset &= value == list_b[index]

        return is_subset

    def _dfs(op_names, op_types, node, pattern):
        if pattern == []:
            return
        start_index = 0
        end_index = len(pattern) - 1
        matched_flag = False
        while start_index <= end_index:
            matched_flag = _validate_input(node.op_type, pattern[end_index])

            if not matched_flag and isinstance(pattern[end_index], tuple):
                end_index -= 1
                continue

            if matched_flag:
                op_names.append(node.name)
                op_types.append(node.op_type)
                break

            return

        if start_index == end_index:
            if matched_flag:
                matched_res = copy.deepcopy(op_names)
                matched_res.reverse()
                op_types_copy = copy.deepcopy(op_types)
                op_types_copy.reverse()
                matched_res.append(op_types_copy)
                if matched_res not in output_result:
                    output_result.append(matched_res)

                op_names.pop()
                op_types.pop()
            return

        for index, value in enumerate(node.input_tensors):
            is_const = False
            # is_const = (value.source_op == [])
            is_const = (isinstance(value.data, np.ndarray))
            if not is_const:
                cur_node = graph.get_node_by_name(value.source_op[0])
                _dfs(op_names, op_types, cur_node, pattern[:end_index])
            if index == len(node.input_tensors) - 1:
                op_names.pop()
                op_types.pop()

    output_result = []

    for v in graph.nodes:
        start_index = len(input_pattern) - 1
        while start_index >= 0:
            find_first_match = _validate_input(v.op_type, input_pattern[start_index])
            if find_first_match:
                break

            if isinstance(input_pattern[start_index], tuple):
                start_index -= 1
                continue

            start_index = -2

        if start_index < 0:
            continue

        visited_op_name = []
        visited_op_types = []

        _dfs(visited_op_name, visited_op_types, v, input_pattern)

    sorted_output = sorted(output_result, key=lambda i: i[-1])

    useless_match_list = []
    for index, value in enumerate(sorted_output):

        if index == len(sorted_output) - 1:
            break

        next_matched_op_names = sorted_output[index + 1][:-1]
        if len(value[:-1]) < len(next_matched_op_names) and \
                _compare_list(value[:-1], next_matched_op_names):
            useless_match_list.append(value)

    for i in useless_match_list:
        sorted_output.remove(i)

    longest_match = {}
    final_output = []
    for i in sorted_output:
        key = i[0]
        if key not in longest_match:
            longest_match[key] = i[-1]
            continue

        if len(longest_match[key]) < len(i[-1]):
            longest_match[key] = i[-1]

    for i in sorted_output:
        if i[0] in longest_match and i[-1] == longest_match[i[0]]:
            final_output.append(i)

    return final_output


def search_pattern(pattern_list, graph):
    """Search the complete pattern in the graph.
    Args:
        pattern_list: a list contains  pattern representation. The pattern representation is also
                      a list and each node in the list is a tuple, its form is like "(op_idx, 
                      op_type)". However, due to a few complicated patterns, they have sub-graph 
                      computation flow. Therefore in a pattern representation, using the fist list
                      represents the main top-down computation flow (from pattern head op to tail
                      op), the left lists represent sub-graphs (their tail nodes must in the main
                      computation flow).
                      e.g:  LayerNorm pattern from bert_large_squad.pb
                      [ [(0, 'Mean'), (1, 'SquaredDifference'), (2, 'Mean'), (3, 'AddV2'), 
                        (4, 'Rsqrt'), (5, 'Mul'), (7 ,'Mul'), (8, 'Sub'), (9, 'AddV2')], 
                        [(5, 'Mul'), (6, 'Mul'), (9, 'AddV2')] ]

        graph: Graph Class, the new graph generated from extractor

    Returns: [string list], as same as search_straight_pattern func.

    NOTE:
        1. The op_idx follows the order in the original frozen model, which means you had better 
           not identify them on your own casually.
        2. the main top-down computation flow follows the "tf control flow". It's a straight chain 
           from start to end in the pattern. Mostly, it has the longest path. Sometimes, this main 
           flow may have sub connection. But you don't need to represent them. A sub-graph must 
           have one op at least that doesn't exist in the main chain. For example, in the above 
           LayerNorm pattern, (0, 'Mean') has a connection with (7 ,'Mul'). But you don't need to 
           represent this relationship.
        3. If a node in sub-graph has several input /output paths, you should split them, each 
           sub-graph has one input /output op. (these ops must be in the pattern).
           For example, the below representtaion should be two sub-graphs:
           Add --- Mul --- Sub
           Add ---^
           [..., [(idx, 'Add'),(idx, 'Mul'),(idx, 'Sub')], [(idx, 'Add'),(id, 'Mul'),
            (idx, 'Sub')], ...]
        4. If a node in sub-graph has several input ops, some of them are from outside. Then you 
           don't need to give the sub-graphs with the outside op.
           For example, the below representtaion should be one sub-graph:
           Add     ---   Mul --- Sub
           outside op ---^
           [..., [(idx, 'Add'),(idx, 'Mul'),(idx, 'Sub')], ...]
        5. If a node in sub-graph just has one input op and this op is from outside, you should 
           use empty tuple () to represents a input op. However, the algorithm doesn't support 
           this kind of pattern. Beause the match result can't make sure the whole connection.
        6. For the symmetric pattern, the sub-graph has consecutive same op type as the main chain 
           (Y or O shape). So these two search results by DFS are duplicated. The algorithm would 
           perform checking before splicing and de-duplication. The sub-graph length <= the main 
           chain length.
        7. Some pattern has several same sub-graphs, these sub-graphs have same tail node and op 
           types are totally same.
           For example:
            a -- b -- c --d -- e --f
                |             |
                c1 -- d1 -----
                |             |
                c2 -- d2 -----
                |             |
                c3 -- d3 -----
            So the splicing step need to check the node name
        For now, the algorithm just support the sub-graph's input /output ops are all in pattern. 
        You can set the sub-graph input as (), but the results need you to check. Mostly, this 
        sub-graph is a part of the pattern.
        As for pattern match / search, apply dfs to every graph list, then check the sub-graph's 
        connection with the main computation flow. The idx would make the returned string list
        with right order.
    """
    # parse the pattern_list and match sub-graph
    def _search_subgraph(subgraph):
        p_subgraph = [c[1] for c in subgraph]
        subgraph_idx = [c[0] for c in subgraph]
        m_subgraph = search_straight_pattern(p_subgraph, graph)
        return (m_subgraph, subgraph_idx)

    # avoid splicing error when sub_graphs with totally same op types and tail node (#7)
    # double check node idx and name
    def _has_duplicated_names_in_main_chain(sub_chain, main_chain, sub_chain_node_idx, has_head):
        main_chain_node_names = [main_chain[i][0] for i in main_chain.keys()]
        ret_flag = False
        if has_head:
            sub_chain_node_names = sub_chain[1:-2]
            sub_chain_node_index = sub_chain_node_idx[1:-1]
        else:
            sub_chain_node_names = sub_chain[:-2]
            sub_chain_node_index = sub_chain_node_idx[:-1]
        for i in range(len(sub_chain_node_names)):
            index = sub_chain_node_index[i]
            if index in main_chain.keys():
                continue
            else:
                name = sub_chain_node_names[i]
                if name in main_chain_node_names:
                    ret_flag = True
                    break
                else:
                    continue

        return ret_flag

    # splicing the main_chain and sub_chain
    def _check_subgraph(iter_ret, m_subgraph, sub_graph_idx, has_head):
        flag = [0] * len(iter_ret)
        for each_sub in m_subgraph:
            for i in range(len(iter_ret)):
                if flag[i] == 0:
                    # get the sub-graph head and tail ops' names
                    tail_idx = sub_graph_idx[-1]
                    tail_name = iter_ret[i][tail_idx][0]
                    if has_head:
                        head_idx = sub_graph_idx[0]
                        head_name = iter_ret[i][head_idx][0]
                        if (each_sub[0] == head_name) and (each_sub[-2] == tail_name):
                            # check sub-graph name (#7)
                            if _has_duplicated_names_in_main_chain(
                                    each_sub, iter_ret[i], sub_graph_idx, has_head):
                                continue
                            for j in range(1, len(each_sub) - 1):
                                iter_ret[i][sub_graph_idx[j]] = [each_sub[j], each_sub[-1][j]]
                                # each sub-graph can be matched more than one main_chain
                            flag[i] = 1
                            # break

                    else:
                        # need to improve the implementation of unknown input node
                        if each_sub[-2] == tail_name:
                            # check sub-graph name (#7)
                            if _has_duplicated_names_in_main_chain(
                                    each_sub, iter_ret[i], sub_graph_idx, has_head):
                                continue
                            for j in range(0, len(each_sub) - 1):
                                iter_ret[i][sub_graph_idx[j]] = [each_sub[j], each_sub[-1][j]]
                                # each sub-graph just can be matched more than one main_chain
                            flag[i] = 1
                            # break
                else:
                    continue
        return iter_ret

    # remove the duplicated results due to the complicated symmetric issues (#6)
    # lists may have same nodes names between each other
    # if has symmetric chains, they must appear consecutively
    # just keep the first one
    def _rm_duplicated_rets(results):
        if len(results) <= 1:
            return results
        keep_index = []
        i = 0
        length = len(results)
        while i < length:
            keep_index.append(i)
            ret_a = set(results[i][:-1])
            i += 1
            start = i
            for j in range(start, length):
                ret_b = set(results[j][:-1])
                if ret_a == ret_b:
                    i += 1
                else:
                    break

        final_results = [results[k] for k in keep_index]
        return final_results

    assert len(pattern_list) > 0, "The input patten_list can not be empty!"
    main_chain = pattern_list[0]
    pattern_length = main_chain[-1][0] + 1
    m_main_chain, main_chain_idx = _search_subgraph(main_chain)
    # if has sub-graph, like LayerNorm
    if len(pattern_list) > 1:
        if m_main_chain == []:
            return m_main_chain
        m_result = []
        iter_ret = []
        # print(len(m_main_chain))
        for v in m_main_chain:
            tmp = {}
            # op.type list
            # tmp[-1] = v[-1]
            # op_idx : [op_name, op_type]
            i = 0
            for idx in main_chain_idx:
                tmp[idx] = [v[i], v[-1][i]]
                i += 1
            assert i == len(v) - 1, "error occurs in dict converting"
            iter_ret.append(tmp)
        # is_symmetric = True  # False
        has_one_no_head = False
        for subgraph in pattern_list[1:]:
            has_head = True
            subgraph_ = copy.deepcopy(subgraph)
            if subgraph[0] == ():
                has_head = False
                has_one_no_head = True
                subgraph_.remove(())
            m_subgraph, subgraph_idx = _search_subgraph(subgraph_)
            if len(m_subgraph) == 0:
                return []
            iter_ret = _check_subgraph(iter_ret, m_subgraph, subgraph_idx, has_head)
        if has_one_no_head:
            logger.warning(
            "Does not completely support this pattern: {} now, please check the output results."\
            .format(pattern_list))
        pattern_max_len = 1
        for each_ret in iter_ret:
            pattern_max_len = max(pattern_max_len, len(each_ret))
        # if pattern is symmetric but the graph has not this pattern
        # for example the pattern just has a half chain of U-shape pattern
        # a--b--c
        # a--b--^    but just has a--b--c
        if pattern_max_len == len(main_chain) or pattern_max_len < pattern_length:
            return []
        for each_ret in iter_ret:
            if len(each_ret) < pattern_max_len:
                continue
            tmp = []
            tmp_op = []
            for k in sorted(each_ret):
                # append name
                tmp.append(each_ret[k][0])
                tmp_op.append(each_ret[k][1])
            # append op_type
            tmp.append(tmp_op)
            m_result.append(tmp)

        # de-duplication due to the symmetric chains
        # take the first one out
        m_result = _rm_duplicated_rets(m_result)
        return m_result

    # if has no sub-graph, like MatMul_BiasAdd
    else:
        return m_main_chain


def construct_node(node_name, op_type, input_tensors=[], output_tensors=[], attr=OrderedDict()):
    """construct node with engine op_type
    Args:
        node_name: string, name of the node
        op_type: string, type of the node
        input_tensors: list, contains the input tensors of the node
        output_tensors: list, contains the output tensors of the node
    Returns:
        new_node: Operator class
    """
    from .ops.op import OPERATORS, Operator
    from .ops.tensor import Tensor
    if op_type in OPERATORS.keys():
        new_node = OPERATORS[op_type]()
        new_node.construct(node_name, op_type, input_tensors=input_tensors,
                           output_tensors=output_tensors, attr=attr)
        return new_node
    else:
        raise ValueError('the {} operation does not support now...'.format(op_type))


def insert_pattern(target_node_names, new_nodes, graph):
    """Replace the specific pattern matched from the new constructed graph with new pattern.
    Args:
        target_node_names: A string list ccontains the names of nodes that will be replaced
        new_nodes: a list contains nodes with Operator class
        graph: The Graph class

    Returns:
        graph: The Graph class which some nodes inside have been replaced.
    """
    # empty node_names
    if len(target_node_names) == 0:
        return graph

    # only one node
    elif len(target_node_names) == 1:
        node_name = target_node_names[0]
        index = graph.get_node_id(node_name)
        graph.remove_nodes([node_name])

    else:
        # check the order
        # in some conditions, not every name in the target_node_names exists in the graph,
        # the node may be removed last time. For example,
        # a--b--c---d0--e0--f0
        #        ---d1--e1--f1
        #        ---d2--e2--f2
        #        ....
        #        ---dn--en--fn
        # the [dn--en--fn] has same op_type, and the algorithm finds n results in graph,
        # but at the first replace iteration, the [a,b,c] has been removed, so the next
        # iterations missing the [a,b,c]. Need to get the real head and tail node names.
        exist_node_index = []
        for i in range(len(target_node_names)):
            try:
                j = graph.get_node_id(target_node_names[i])
                exist_node_index.append([i, j])
            except BaseException:
                continue
        exist_node_index = sorted(exist_node_index, key=lambda x: x[1])
        exist_node_names = [target_node_names[i[0]] for i in exist_node_index]

        head_name = exist_node_names[0]
        tail_name = exist_node_names[-1]
        head_id = graph.get_node_id(head_name)
        tail_id = graph.get_node_id(tail_name)
        # in the graph.nodes[head_id:tail_id+1], there may be some other nodes
        # have input tensors of new_node
        index = head_id
        i = 0

        while i < len(exist_node_names):
            if exist_node_names[i] == graph.nodes[index].name:
                graph.remove_nodes([exist_node_names[i]])
                i += 1
            else:
                # if not has extra input tensors
                if len(exist_node_names) == (tail_id - head_id + 1):
                    raise ValueError("The target nodes have node {} while graph has node {}."\
                    .format(exist_node_names[i], graph.nodes[index].name))
                # if has extra input tensors
                else:
                    index += 1
    # insert new_nodes

    graph.insert_nodes(index, new_nodes)

    return graph


def pattern_mapping(mapping_dict, graph):
    """
    Args:
        mapping_dict: a element in mapping_config[pattern_name], config for pattern mapping.
        graph: Graph class.
    Returns:
        tuple, the first element is the new nodes insert start idx, the second element is a new 
        node list, the third is a list contains required old nodes need to be returned from origin 
        pattern.

    A example of mapping_config:
    mapping_config: {'pattern_name':[
                   {'patterns': {'in': [(0, 'Reshape), ...], 'out':[(0, 'PaddingSequence')]},
                    'search_mode': op_type
                    'node_names': {0: 'embeddings/reshape', 1: 0, ...},
                    'input_tensors': {0:[{0:[0]}, [[0], 1]], 1:[{1:[0], 2:[1,2]},[[0,1], 3]], 
                                      2:[{},[[],1]], ..., m:[{'input_data':[1]}, [[0],1]},
                    'output_tensors': {2:[{0:[0]}, [[0],1]], ...},
                    'returns': [0, 1, 2],
                   },                         # one representation of this pattern
                   {...},                     # another representation of this pattern
                    ...
    ]}

    'pattern_name': the name of the customized pattern representation, for example, 'LayerNorm'.
    'patterns': give the pattern representations before ('in') and after ('out') fusion. See the 
                search_pattern() function for more details about pattern representation.
    'search_mode': 'op_type' or 'node_name'. If set it as op_type, the algorithm will search 
                in_pattern in graph. If set node_name, means in_pattern is just representing the 
                search result. For example:
                in_pattern is [[(0, 'input_ids'), (1, 'segment_ids'), (2, 'input_mask')]]
                out_pattern is [[(0, 'Input')]]
    'node_names': set node name for each node in pattern after fusion. Key means the node idx, 
                the value must be string or int (idx). If the value is the string, just use it as 
                the node's name. If the value is the idx, use the name of idx-th node in the 
                pattern berfore fusion. If the in_pattern has n match_results in the graph, it 
                will add "_n" after the name, for example, the new node name should be 
                "embeddings/reshape_0" after mapping of the first match_result.
    'input_tensors': the input_tensors of patterns before or after fusion should be same. The key 
                in the dict is the idx of the new node, and the first dict in the value list means 
                where this tensor get from the pattern before fusion, and the second means where 
                this tensor go to the pattern after fusion. For example, in '0:[{0:[0]}, 
                [[0], 1]]', '0' in the key means it's the first new node in out_pattern, '{0:[0]}'
                means the tensor is the first tensor of the first node in in_pattern, '[[0], 1]' 
                means the first new node's first input_tensor is the tensor and this node has 
                total 1 input_tensor. So the first element in the value gives the source info of 
                input_tensors, the second gives the dest info of the input_tensors.However, 
                sometimes source info has the form like '{1:[0], 2:[1,2]}', the '[1,2]' means the 
                idx of tensor is not sure, maybe 1 or 2. It will happens to some sepcial op, like 
                'BiasAdd', its 'bias' tensor maybe in unfixed location. If some input_tensors only
                can get from other node outside the pattern, you can just specify it by give the 
                node name in graph.
    'output_tensors': the output_tensors of patterns before or after fusion should be same. The 
                representtaion is same meaning of 'input_tensors'.
    'returns': set the node idx, and return the idx-th node of pattern before fusion. Sometimes 
                need these nodes for writing node attributes in pattern after fusion. If don't 
                need return, set the value as [].

    Note that the pattern after fusion (n->n / n->1)is must be sequence pattern or just separated 
    from each other, like [a->b->c->d->e], or [a] / [a,b,c,d,e]. That means if one pattern is too 
    complicated, or the pattern after fusion is too complicated, you had better decompose it.

    """

    # search pattern and get the in_pattern match_result and info for out_pattern
    def _get_pattern_info():
        in_pattern = mapping_dict['patterns']['in']
        search_mode = mapping_dict['search_mode']
        assert search_mode in ['op_type', 'node_name'], 'Unsupported mode'
        in_match_result = []
        if search_mode == 'op_type':
            in_match_result = search_pattern(in_pattern, graph)
        else:
            # check whether the nodes exit or not
            nodes_exist = True
            in_result = []
            for n in in_pattern[0]:
                n_name = n[1]
                try:
                    _ = graph.get_node_id(n_name)
                    in_result.append(n_name)
                except BaseException:
                    nodes_exist = False
                    break
            if nodes_exist:
                in_result.append([])
                in_match_result.append(in_result)
            else:
                in_match_result = []
        num_match = len(in_match_result)

        new_node_names = []
        name_reference = mapping_dict['node_names']
        input_tensors = []
        input_tensors_reference = mapping_dict['input_tensors']
        output_tensors = []
        output_tensors_reference = mapping_dict['output_tensors']
        ret_old_nodes = []
        returns_inference = mapping_dict['returns']
        for i in range(num_match):
            # get the node_names in out_pattern
            names = []
            for j in range(len(name_reference)):
                tmp = name_reference[j]
                if isinstance(tmp, int):
                    names.append(in_match_result[i][tmp])
                elif isinstance(tmp, str):
                    if num_match == 1:
                        names.append(tmp)
                    else:
                        names.append(tmp + '_' + str(i))
                else:
                    raise ValueError(
                        'Do not support the setted node_names types,it must be int or str,'\
                            'rather than {}.'.format(type(tmp)))
            new_node_names.append(names)

            # get the input_tensors in out_pattern
            in_tensors = []
            for k in range(len(input_tensors_reference)):
                source = input_tensors_reference[k][0]
                dest = input_tensors_reference[k][1]
                tmp = []
                for kv in source:
                    if len(kv) != 0:
                        tmp_k = list(kv.keys())[0]
                        tmp_v = kv[tmp_k]
                        if isinstance(tmp_k, int):
                            node = graph.get_node_by_name(in_match_result[i][tmp_k])
                            assert len(tmp_v) <= 2
                            if len(tmp_v) == 1:
                                v = tmp_v[0]
                            # for support the bias_add, the bias tensor idx may be not always same
                            else:
                                v = tmp_v[0]
                                t0 = node.input_tensors[v]
                                pre_node = graph.get_node_by_name(in_match_result[i][tmp_k - 1])
                                if t0.name == pre_node.output_tensors[0].name:
                                    v = tmp_v[1]
                            tmp.append(copy.deepcopy(node.input_tensors[v]))
                        elif isinstance(tmp_k, str):
                            node = graph.get_node_by_name(tmp_k)
                            tmp.append(copy.deepcopy(node.output_tensors[tmp_v[0]]))
                        else:
                            raise ValueError(
                                'Do not support the setted input_tensors types,'\
                                    'it must be int or str, rather than {}.'.format(type(tmp_k)))

                in_tensors.append([tmp, dest])
            input_tensors.append(in_tensors)

            # get the output_tensors in out_pattern
            out_tensors = []
            for m in range(len(output_tensors_reference)):
                source = output_tensors_reference[m][0]
                dest = output_tensors_reference[m][1]
                tmp = []
                for kv in source:
                    if len(kv) != 0:
                        tmp_k = list(kv.keys())[0]
                        tmp_v = kv[tmp_k]
                        if isinstance(tmp_k, int):
                            node = graph.get_node_by_name(in_match_result[i][tmp_k])
                        else:
                            raise ValueError(
                                'Do not support the setted output_tensors types, it must be int,'\
                                    'rather than {}.'.format(type(tmp_k)))
                        assert len(tmp_v) == 1, 'Output tensor must be specified.'
                        tmp.append(copy.deepcopy(node.output_tensors[tmp_v[0]]))
                out_tensors.append([tmp, dest])
            output_tensors.append(out_tensors)
            # get the returns node in in_pattern for set the attr of node in out_pattern later
            ret_tmp = []
            for idx in returns_inference:
                node = graph.get_node_by_name(in_match_result[i][idx])
                ret_tmp.append(copy.deepcopy(node))
            ret_old_nodes.append(ret_tmp)

        return (in_match_result, new_node_names, input_tensors, output_tensors, ret_old_nodes)

    # created the new nodes in out_pattern

    def _create_out_pattern(new_node_names, input_tensors_list, output_tensors_list):
        from .ops.tensor import Tensor
        out_pattern = mapping_dict['patterns']['out']
        # sequence = mapping_dict['sequence']
        sequence = True if len(out_pattern[0]) > 1 else False
        num_pattern = len(new_node_names)
        new_node_types = []
        new_nodes = []
        for pattern in out_pattern:
            for p in pattern:
                new_node_types.append(p[1])
        # single sequence pattern or several separated nodes
        for i in range(num_pattern):
            one_p = []
            pre_node = None
            for j in range(len(new_node_types)):
                op_type = new_node_types[j]
                node_name = new_node_names[i][j]
                # set the input_tensors
                tensors = input_tensors_list[i][j][0]
                idx_tensors = input_tensors_list[i][j][1][0]
                num_tensors = input_tensors_list[i][j][1][1]
                input_tensors = [0] * num_tensors
                if pre_node is None:
                    assert num_tensors == len(tensors),\
                        'No pre node of {}, please supply the completed input_tensors.'.format(
                            node_name)
                    input_tensors = tensors
                else:
                    assert num_tensors == len(tensors) + 1,\
                        'Only support the sequence pattern for now.'
                    for k in range(len(idx_tensors)):
                        input_tensors[idx_tensors[k]] = tensors[k]
                    input_tensors[input_tensors.index(0)] = copy.deepcopy(
                        pre_node.output_tensors[0])
                # set the output_tensors
                tensors = output_tensors_list[i][j][0]
                idx_tensors = output_tensors_list[i][j][1][0]
                num_tensors = output_tensors_list[i][j][1][1]
                # output_tensors = [0] * num_tensors
                # separated node without connections with other nodes in out_pattern
                if len(tensors) != 0:
                    assert len(tensors) == num_tensors
                    output_tensors = tensors
                # has connections
                else:
                    assert num_tensors == 1, 'Only support op with one output_tensor for now.'
                    output_tensors = [Tensor(name=node_name + ':0',
                                             source_op=[node_name],
                                             dest_op=[new_node_names[i][j + 1]])]

                new_node = construct_node(
                    node_name,
                    op_type,
                    input_tensors=input_tensors,
                    output_tensors=output_tensors)
                one_p.append(new_node)

                if sequence:
                    pre_node = new_node

            new_nodes.append(one_p)

        return new_nodes

    # replace the in_pattern with out_pattern
    def _replace_pattern(in_match_result, new_nodes, graph):
        assert len(in_match_result) == len(
            new_nodes), 'out_pattern should have as some num as in_pattern in graph.'
        for i in range(len(in_match_result)):
            each_ret = in_match_result[i][:-1]
            insert_nodes = new_nodes[i]
            graph = insert_pattern(each_ret, insert_nodes, graph)

        return graph

    # 1. check the format of mapping_dict 
    mapping_dict = pattern_mapping_conf_validation(mapping_dict)
    # 2. get the necessary info for out_pattern construction
    in_match_result, new_node_names, input_tensors_list, output_tensors_list, \
        ret_old_nodes = _get_pattern_info()
    # 3. create the nodes in out_pattern
    new_nodes = _create_out_pattern(new_node_names, input_tensors_list, output_tensors_list)
    # 4. remove the nodes in in_pattern, and insert the nodes in out_pattern
    graph = _replace_pattern(in_match_result, new_nodes, graph)
    # 5. return graph after pattern mapping
    return (graph, new_node_names, ret_old_nodes)


def list2str(src_perm):
    """convert the shape list to str for emitting yaml
    Args:
        src_perm: list, for example [1,2,3,4]
    Returns:
        ret: str, for example '1,2,3,4'
    """
    ret = ','.join(str(i) for i in list(src_perm))
    return ret


def str2list(src_str):
    """convert the str to shape list
    Args:
        src_str: for example '1,2,3,4'
    Returns:
        ret: list, for example [1,2,3,4]
    """
    ret = []
    s_list = src_str.split(',')
    ret = [int(i) for i in s_list]
    return ret

def pattern_mapping_conf_validation(conf_dict):
    dict_schema = Schema({
    'patterns' : Schema({
        'in' : And(
            list,
            lambda s: all(Schema([(), (int, Or(str, Schema([str])))]).validate(p) for p in s),
            lambda s: all((len(t)==2 for t in p[1:]) for p in s[1:]),
            error='The in pattern must supply the node index and op_type, and only the head node'\
            'in sub-chain can be empty.'
        ),
        'out': Schema([[(int, str)]], error='The out pattern must be straight chain.') 
    }),

    'search_mode': Or('op_type', 'node_name', error='Only support op_type or node_name these '\
                        'two modes'),

    'node_names': Schema({
        int: Or(int, str)
    },  error='For node_names, the key is the out node index while the value is the old node ' \
                'index or a specific name'),

    'input_tensors': Schema({
        int: And(
            list,
            lambda s: And(all(list for i in s), len(s)==2, error='The value in input_tensors ' \
            'must be list and has length 2'),
            lambda s: Or(
                Schema([{Or(int, str): Schema([int])}]).validate(s[0]),
                And(len(s[0])==0, isinstance(s[0], list),
                error='The first element can be empty list or contains several dict for telling'\
                ' where can get the input_tensor of new_node'),
            lambda s: Or(
                Schema([[int], int]).validate(s[1]),
                Schema([[], int]).validate(s[1],
                error='The second element can be empty list or contains several int number for'\
                ' telling the input_tensor index of new_node')
            )
        ))
    }),

    'output_tensors': Schema({
        int: And(
            list,
            lambda s: And(all(list for i in s), len(s)==2, error='The value in output_tensors ' \
            'must be list and has length 2'),
            lambda s: Or(
                Schema([{int: Schema([int])}]).validate(s[0]),
                And(len(s[0])==0, isinstance(s[0], list),
                error='The first element can be empty list or contains a dict for telling'\
                ' where can get the output_tensor of new_node'),
            lambda s: Or(
                Schema([[int], int]).validate(s[1]),
                Schema([[], int]).validate(s[1],
                error='The second element can be empty list or contains int number for'\
                ' telling the output_tensor index of new_node')
            )
        ))
    }),

    'returns': Or(Schema([]), Schema([int]), error='Returns can be empty list or contains some ' \
     'old node index for set attr of new node later'),

    },  ignore_extra_keys=True)

    return dict_schema.validate(conf_dict)