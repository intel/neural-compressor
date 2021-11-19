# Graph Fusion and Optimization

The main purpose of graph fusion and optimization is to simplify network and speed up inference. We take this process as a way of pattern mapping.  User could call the `pattern_mapping` API in [`engine.compile.graph_utils`](https://github.com/intel/neural-compressor/blob/master/engine/compile/graph_utils.py) to realize it. The process contains three steps: **1. obtain the necessary information for new pattern construction; 2. create nodes and establish connections; 3. remove old pattern and insert new pattern.**  Before implementing graph fusion and optimization, user should supply a config (dict) to instruct pattern mapping.

## Pattern mapping dict

Here is a example of pattern mapping dict (a LayerNorm pattern in doc [add_customized_pattern](https://github.com/intel/neural-compressor/tree/master/engine/docs/add_customized_pattern.md)): 

```python
{    'patterns': {
                 'in': [[(0, 'ReduceMean'), (1, 'Sub'), (2, 'Pow'), (3, 'ReduceMean'), (4, 'Add'), (5, 'Sqrt'), (6, 'Div'), (7,'Mul'), (8, 'Add')]],
                 'out': [[(0, 'LayerNorm')]]
                 },
     'search_mode': 'op_type',
     'node_names': {
                    0: 8
                   },
     'input_tensors': {
                        0: [[{
                            0: [0]
                        }, {
                            7: [1]
                        }, {
                            8: [1]
                        }], [[0, 1, 2], 3]]
                       },
     'output_tensors': {
                        0: [[{
                            8: [0]
                        }], [[0], 1]]
                    },
     'returns': [4]
     }
```

- `patterns` : gives the patterns representations before (`in`) and after (`out`) mapping. It is a dict with two keys (`in` and `out`), and the values are the corresponding pattern representations. About the representation, refer to [pattern_recognition](https://github.com/intel/neural-compressor/tree/master/engine/docs/pattern_recognition.md).

- `search_mode`:  a string must be `op_type` or `node_name`. If set as `op_type`, the algorithm will search the old pattern (`in`) in graph. If set as  `node_name`, just means the old pattern (`in`) is  representing the search result. In most conditions, it should be `op_type`.

- `node_names`:  a dict for setting node name for each node in new pattern. The key is the node index of new pattern (`out`).  The value must be string or integer. If string, just use it as the node's name. If integer, use the name of index-th (`integer`) node in the old pattern(`in`).  Usually, the old pattern has many match_results in the graph, the algorithm would add "_n" after the name when the value is string. For example, the new node `0` (`0: embeddings_reshape `) name should be `embeddings/reshape_0` of the first match_result.

- `input_tensors`: a dict for telling where to get the input tensors of new pattern (`out`).  These input tensors of patterns before or after fusion should be same.  The input tensors we said here are aimed at pattern. For example, the pattern can be thought as a big node and it receives and emits tensors which should be equal before or after fusion. So the `input_tensors` supplies information of the input tensor outside rather than the tensors inside pattern.  The key is the node index of new pattern (`out`). The first list in the value list means where are the outside input tensors from (which input tensors index of which node in old pattern). And the second list means which locations these tensors would be put (which input tensor index of node in new pattern and number of them ) . For example, the above `0 :[ [{0: [0]}, {7: [1]}, {8:[1]}], [[0, 1, 2], 3] ]` shows that the 0-th node in new pattern needs three outside input tensors. The first one is 0-th input tensor of 0-th node in old pattern, the second one is 1-st input tensor of 7-th node in old pattern, and the third one is 1-st input tensor of 8-th node in old pattern. `[[0, 1, 2], 3]` means these three tensors are as 0-th, 1-st, 2-nd input tensor respectively of 0-th node in new pattern (just copy tensors) and this node has three input tensors totally. Sometimes the tensor number is greater than the length of first list for the reason that this node would receive tensor from its last node in new pattern, like `[[0, 1], 3]`. This 2-nd tensor will be generated and sent automatically inside new sequence pattern. 

  > NOTE:
  >
  > - In some models, the tensor index in `input_tensors` could be different in a pattern. For example, pattern `[[(0, 'MatMul'), (1, 'BiasAdd'), (2, 'Add')]]` , the `(2, 'Add')` node have 2 input tensors, the one is from `(1, 'BiasAdd')`, the another one is from other node, which is need to be written into `input_tensors`. However, this tensor index might be 0 in part of matched results while 1 in other results.  You can use `{2: [0, 1]}` to point this phenomenon and the algorithm would check the tensor name and get the correct one.
  > -  If some input_tensors only can get from other node outside the pattern, you can just specify it by give the node name in graph, like `0: [[{'reshape_0': [0]}], [[0], 1]]`.

- `output_tensors`: a dict for telling where to get the output tensors of new pattern (`out`). Its representations are similar to `input_tensors`. In most situations, the last nod of new pattern should has the same output tensors as the last node of old pattern (sequence pattern for now and all nodes have one output tensor). So, usually users just need set the last node output tensors and leave the rest nodes output tensors empty while keeping the number of tensors (the algorithm would generate internally).  For example, a new pattern is `[[(0, 'op_type'), (1, 'op_type'), (2, 'op_type')]]`, then its `output_tensors` could be like `{0: [[], [[], 1]], 1: [[], [[], 1]], 2: [[{old_node_index: [0]}], [[0], 1]]}`

- `returns`: a list contains some nodes indexes of old pattern (`in`). It would return these old nodes when pattern mapping finish.  Sometimes user requires specific nodes for setting node attributes in new pattern. You can set the value as `[]` if no need.

## Obtain the necessary information for new pattern construction

The `pattern_mapping` API first search the `in` pattern and get the matched result. Then prepare node names, input tensors, output tensors these necessary information for `out` pattern by following the above pattern mapping dict. It also store some specific nodes in old pattern if needed.  These related variables have the same length as matched result of `in` pattern. For more details, please see the `_get_pattern_info` function in [`engine.compile.graph_utils`](https://github.com/intel/neural-compressor/blob/master/engine/compile/graph_utils.py).

## Create nodes and establish connections

In this step, `pattern_mapping` API receives the above node names, input tensors and output tensors information to create new nodes in `out` pattern. The main part is to establish nodes connections by filling and constructing tensors. The connections outside are maintained with keeping the tensors in or out patterns same. As for the connections inside new pattern, output tensor of pre-node would be generated automatically and become one input tensor of next node due to sequential flow. For more details, please see the `_create_out_pattern` function in [`engine.compile.graph_utils`](https://github.com/intel/neural-compressor/blob/master/engine/compile/graph_utils.py).

## Remove old pattern and insert new pattern

After creating new pattern, the final step is to replace the old pattern with new pattern. For more details, please see the `_replace_pattern` function in [`engine.compile.graph_utils`](https://github.com/intel/neural-compressor/blob/master/engine/compile/graph_utils.py).


Do not forget to set specific attributes of your new pattern if it has. These attributes could influence the inference of engine. And the `returns` supply the way to obtain them from the nodes in old pattern. Please see the [add_customized_pattern](https://github.com/intel/neural-compressor/tree/master/engine/docs/add_customized_pattern.md) doc for adding your own example.
