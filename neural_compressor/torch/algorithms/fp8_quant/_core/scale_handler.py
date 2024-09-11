from .._quant_common.quant_config import ScaleFormat
from torch import Tensor, nn


def create_scale_tensor(orig_tensor, scale_format):
    if scale_format == ScaleFormat.CONST:
        return nn.Parameter(orig_tensor)
    elif scale_format == ScaleFormat.SCALAR:
        return scale_to_scalar(orig_tensor)
    else:
        raise ValueError("unexpected scale format value {}".format(scale_format))


# scalar scale is a performance optimization for LLM layers in small BS
def scale_to_scalar(scale):
    if isinstance(scale, Tensor):  # tensor case
        if scale.numel() == 1:
            return scale.item()
        else:
            raise Exception("scale as scalar isn't supported for scale tensors of dim > 0")
    elif isinstance(scale, float):  # already scalar case
        return scale
    else:
        raise Exception("unexpected scale instance type, expected Torch.tensor or float number")
