import transformers

from .utils import torch, nn

class WrappedGPT:
    """This class wraps a GPT layer for specific operations."""

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        if isinstance(self.layer, transformers.Conv1D):
            self.rows, self.columns = self.columns, self.rows

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.sum_metric_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        # for DsnoT
        self.mean = torch.zeros((self.columns), device=self.dev)
        self.var = torch.zeros((self.columns), device=self.dev)
        self.ntokens = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        import torch

        if isinstance(self.layer, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.sum_metric_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)

        mean_inp = torch.mean(inp, dim=1, keepdim=True)

        var_inp = torch.var(inp, dim=1, unbiased=False, keepdim=True)
        num_inp = inp.shape[1]
        self.var = var_inp if self.ntokens == 0 else (self.var * self.ntokens + var_inp * num_inp) / (self.ntokens + num_inp)
        self.mean = mean_inp if self.ntokens == 0 else (self.mean * self.ntokens + mean_inp * num_inp) / (self.ntokens + num_inp)
        self.ntokens += num_inp

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        self.sum_metric_row += torch.sum(inp, dim=1) / self.nsamples