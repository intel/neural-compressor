from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


def show_nest_dict(result, depth=1):
    indent = "***-"*depth
    if not isinstance(result, dict):
        
        print(f"[{depth}]{indent}: {result}")
    else:
        for key, val in result.items():
            print(f"[{depth}]{indent}: {key}")
            show_nest_dict(val, depth=depth+1)


def traverse_model(model, prefix="", depth=1, result={}, key = 0, ops=[]):
    module_lst =list(model.named_children())
    if len(module_lst) == 0:
        # layer name: 'encoder.layer.7.attention.self.query'
        # model repr: Linear(in_features=768, out_features=768, bias=True)
        # class name: 'Linear'
        result[key] = (prefix, model, model.__class__.__name__)
    for i, (name, sub_module) in enumerate(module_lst, 1):
        indent = "    "*depth
        new_name = prefix + '.' + name if prefix != "" else name
        model_type = sub_module.__class__.__name__
        print(f"Depth: [{depth}]",indent, f"[{model_type}]{ new_name}")
        sub_key = (depth, i, model_type)
        if sub_key not in result[key]:
            result[key][sub_key] = dict()
        traverse_model(sub_module, prefix=new_name, depth=depth+1, result=result[key], key = sub_key, ops=ops)

from typing import Dict, List

def get_depth(d) -> int:
    """Query the depth of the dict.

    Args:
        result: _description_

    Returns:
        _description_
    """
    if isinstance(d, dict):
        return 1 + max(get_depth(v) for v in d.values())
    return 0
    

def get_dict_at_depth(d, target_depth, result, depth=0):
    if depth == target_depth:
        result.append(d)
        return
    elif depth < target_depth and isinstance(d, dict):
        for k, v in d.items():
            get_dict_at_depth(v, target_depth, result, depth=depth+1)

def get_element_under_depth(d, ops_lst):
    if isinstance(d, dict):
        for k, v in d.items():
            get_element_under_depth(v, ops_lst)
    else:
        ops_lst.append(d)

def collect_block(depth_block):
    collect_result = []
    cnt = 0
    for i, block in enumerate(depth_block, 1):
        print(f"Collected block: {block}")
        show_nest_dict(block)
        ops_lst = []
        get_element_under_depth(block, ops_lst)
        filter_lst = [k for k in ops_lst if k[2] == "Linear"]
        if len(filter_lst) >= 3:
            cnt += 1
            print(cnt, i, [(k[0]) for k in filter_lst])
            collect_result.append(filter_lst)
    return collect_result

def show_block(attention_block):
    for i, block in enumerate(attention_block, 0):
        print(f"BLOCK[{i}], {block}")

def get_block(model):
    op_positions = {0: dict()}
    traverse_model(model, result=op_positions)
    # get the max depth of the result
    max_depth = get_depth(op_positions)
    attention_depth = max_depth - 2
    depth_block_lst= []
    # collect all block with specified depth
    get_dict_at_depth(op_positions, attention_depth, depth_block_lst, 0)
    # collect ops within block
    attention_block = collect_block(depth_block_lst)
    show_block(attention_block)
    return attention_block






# test
test = 0
if test:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    attention_block = get_block(model)