import torch
from collections import UserDict
from packaging.version import Version
from neural_compressor.torch.utils import get_torch_version

def get_example_inputs(model, dataloader):
    version = get_torch_version()
    from neural_compressor.torch.algorithms.smooth_quant import move_input_to_device

    # Suggest set dataloader like calib_dataloader
    if dataloader is None:
        return None
    device = next(model.parameters()).device
    try:
        for idx, (input, label) in enumerate(dataloader):
            input = move_input_to_device(input, device)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, (list, tuple)):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    except Exception as e:  # pragma: no cover
        for idx, input in enumerate(dataloader):
            input = move_input_to_device(input, device)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, list) or isinstance(input, tuple):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    if idx == 0:
        assert False, "Please checkout the example_inputs format."
