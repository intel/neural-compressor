# The design copied from
# https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py.



import torch


class Accelerator:
    def empty_cache(self):
        pass


class CUDA_Accelerator(Accelerator):
    def __init__(self) -> None:
        pass

    def memory_allocated(self):
        pass

    def max_memory_allocated(self):
        pass

    def memory_reserved(self):
        pass

    def max_memory_reserved(self):
        pass

    def reset_peak_memory_stats(self):
        pass

    def cuda(self):
        pass

    def empty_cache(self):
        torch.cuda.empty_cache()


cuda_accelerator = CUDA_Accelerator()


def auto_detect_accelerator():
    return cuda_accelerator