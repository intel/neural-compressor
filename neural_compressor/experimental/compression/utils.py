from functools import partial
try:
    from ...utils.utility import LazyImport
    from neural_compressor.utils import logger
    LazyImport('torch.nn')
    torch = LazyImport('torch')
    nn = torch.nn
    tf = LazyImport('')
except:
    import torch
    import torch.nn as nn
    import tensorflow
    import logging
    logger = logging.getLogger(__name__)


def get_layers(model):
    """get each layer's name and its module
    Args:
        model: The model to be pruned.

    Returns: each layer's name and its modules
    """
    layers = []
    search_flag = False
    def unfoldLayer(module):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """
        nonlocal search_flag
        nonlocal layers
        if search_flag:
            return
        if hasattr(type(module),"__name__") and 'ModuleList' in type(module).__name__:
            layers = module
            search_flag = True
        layer_list = list(module.named_children())
        for item in layer_list:
            module = item[1]
            if isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers

@torch.no_grad()
def collect_layer_inputs(model, layers, layer_idx, prev_inputs, device='cuda:0'):
    """
    attention_flag: If True collect attention_mask list else the auto-genated causal_attention_mask.
    device: Specify the type of device to return.
    """
    inputs = []
    model_dev = model.device
    attention_mask = None
    # 'alibi' is a necessary attribute for the bloom models
    inputs_info = {'attention_mask': None}
    model_type = model.config.model_type
    if 'bloom' in model_type:
        inputs_info['alibi'] = None
    if layer_idx == 0:
        layer = layers[layer_idx]
        def forward(self, hidden_states, **kwargs):
            # inputs[inputs_info['idx']] = input_ids # TODO solve the problem of batchsize!=1
            inputs.append(hidden_states.to(device))
            inputs_info['attention_mask'] = kwargs['attention_mask']
            if 'alibi' in kwargs.keys():
                inputs_info['alibi'] = kwargs['alibi']
            raise ValueError
        
        forward_cache = layers[layer_idx].forward
        layer.forward = partial(forward, layer)
        for batch in prev_inputs:
            try:
                hidden_states = list(batch.values())[0].to(model_dev)
                model(hidden_states)
                # model(**batch)
            except ValueError:
                pass
        layer.forward = forward_cache
        for key in inputs_info.keys():
            if inputs_info[key] is not None:
                inputs_info[key] = inputs_info[key].to(device)
    else:
        prev_layer = layers[layer_idx-1]
        
        for batch in prev_inputs:
            prev_output = prev_layer(*batch) #需要注意调用前先设置好prev_mask
            batch[0] = prev_output[0]
            inputs.append(batch)
            
    return inputs, inputs_info

