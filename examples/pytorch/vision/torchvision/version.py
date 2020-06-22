__version__ = '0.6.0a0+b68adcf'
git_version = 'b68adcf9a9280aef02fc08daed170d74d0892361'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
