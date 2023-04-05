BLOCK_PATTERNS = [
    # ['OP_TYPE', NUM_OPS]

    [['Linear', 4], ['Linear', 4]], # 
    [['Linear', 4], ['Linear', 3]], # Llama
    [['Linear', 4], ['Linear', 2]], # T5-Encoder, OPT
    [['Linear', 2], ['Linear', 2]], # 
    [['Conv1D', 2], ['Conv1D', 2]], # GPT-2
    [['Linear', 4], ['Linear', 1], ['Linear', 1]], # Bert 
    [['Linear', 4], ['Linear', 4], ['Linear', 2]],  # T5-Decoder
]


from typing import Dict, List

def traverse_model(model, prefix="", depth=1, result={0: {}}, key = 0, ops=[]):
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

def get_depth(d) -> int:
    """Query the depth of the dict."""
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

def search_pattern(pos_info, pattern):
    max_depth = get_depth(pos_info)
    matched_cnt = 0
    result = []
    for depth in range(max_depth, -1, -1):
        attention_depth = depth
        depth_block_lst = []
        get_dict_at_depth(pos_info, attention_depth, depth_block_lst, 0)
        target_op_types = set(pair[0] for pair in pattern)
        for i, block in enumerate(depth_block_lst):
            sub_block_lst = []
            get_dict_at_depth(block, 1, sub_block_lst, 0)
            block_pattern = []
            block_result = []
            for sub_block in sub_block_lst:
                ops_lst = []
                get_element_under_depth(sub_block, ops_lst)
                filter_ops = [op for op in ops_lst if op[2] in target_op_types]
                if len(filter_ops) > 0:
                    sub_block_pattern = [filter_ops[0][2], len(filter_ops)]
                    block_pattern.append(sub_block_pattern)
                    ops_name = [op[0] for op in filter_ops]
                    block_result.append(ops_name)
            if block_pattern == pattern:
                matched_cnt += 1
                print(f"[DEPTH] {depth} [BLOCK] {i},  Found block match pattern {pattern}!!")
                print(f"[Block keys] {block.keys()}")
                print(f"[Block Ops] { [pair[0] for pair in ops_lst if pair[2] in target_op_types]}")
                result.append(block_result)
    if matched_cnt > 0:
        print(f" Found {matched_cnt} blocks")
    return matched_cnt, result


def get_ffn_block(detect_result):
    """Collect ffn blocks from detect result."""
    import itertools
    ffn_block_lst = []
    for block_lst, pattern in detect_result:
        for block in block_lst:
            ffn_block = list(itertools.chain(*block[1:]))
            if ffn_block:
                ffn_block_lst.append(ffn_block)
    return ffn_block_lst

def get_attention_block(detect_result):
    attention_block = []
    for block_lst, pattern in detect_result:
        for block in block_lst:
            if block:
                attention_block.append(block[0])
    return attention_block

def get_blocks(fused_model):
    pos_info = {0: {}}
    traverse_model(fused_model, result=pos_info)
    # [([[[sub_block1, sub_block2, ...]], [], ], pattern)]
    detect_result = [] 
    for pattern in BLOCK_PATTERNS:
        matched_cnt, result = search_pattern(pos_info, pattern)
        if result:
            detect_result.append((result, pattern))
    ffn_block_lst = get_ffn_block(detect_result)
    attention_block_lst = get_attention_block(detect_result)
    print(f"FFN BLOCKS: {ffn_block_lst}")
    print(f"Attention BLOCKS: {attention_block_lst}")
    return ffn_block_lst, attention_block_lst

def merge_with_cap(block_lst, cap):
    """Filter ops according to capability.
    step1, filter ops
    step2, assign op type

    Args:
        block_lst: _description_
        cap: _description_
    """
    cap_ops = cap.get('op_wise', {})
    ops_info = cap_ops.keys()
    filter_result = []
    for block in block_lst:
        op_info_lst = [pair for pair in ops_info if pair[0] in block]
        filter_result.append(op_info_lst)
    return filter_result



# test
test = 1
if test:
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    ffn_block_lst, attention_block_lst = get_blocks(model)
