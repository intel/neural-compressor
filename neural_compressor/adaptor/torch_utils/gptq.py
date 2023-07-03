import math
import time

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

# different models may have different input structures. 
# arch_inputs = {
#     'opt': ['attention_mask'],
#     'bloom': ['attention_mask', 'alibi'],
#     'llama': ['attention_mask', 'position_ids']
# }

DEBUG = False 

def is_leaf(module):
    children_cnt = 0
    for n in module.children():
        children_cnt += 1
    return True if children_cnt == 0 else False

def trace_gptq_target_blocks(module, module_types = [torch.nn.ModuleList]):
    """
    seperately trace two parts related to gptq:
    1. embedding layers (layers before transformer stacks)
    2. transformer stacks
    """
    gptq_related_blocks = {
        "embeddings": {},
        "transformers_pre": {}, # todo
        "transformers_name": None,
        "transformers": None,
        "transformers": {}, # todo
    }
    for n, m in module.named_modules():
        if type(m) in module_types:
            gptq_related_blocks["transformers_name"] = n
            gptq_related_blocks["transformers"] = m
            return gptq_related_blocks
        else:
            if is_leaf(m):
                gptq_related_blocks["embeddings"][n] = m
    return gptq_related_blocks

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)
    
class InputHooker(nn.Module):
    def __init__(self, module, arch, inp, cache):
        super().__init__()
        self.module = module
        self.inp = inp
        self.cache = cache
        # self.input_args = arch_inputs[arch]
    def forward(self, inp, **kwargs):
        # import pdb;pdb.set_trace()
        self.inp[self.cache['i']] = inp
        self.cache['i'] += 1
        for arg in kwargs:
            if isinstance(kwargs[arg], torch.Tensor):
                self.cache[arg] = kwargs[arg]
            else:
                continue
        raise ValueError

class GPTQuantizer(object):
    """Main API"""
    def __init__(self, model, dataloader, device, args):
        # find the stacked transformers module
        # generally, an llm models follows such structure:
        # embeddings => [transformer_block_1, transformer_block_2, ...] => lm_head
        self.model = model
        self.use_cache = model.config.use_cache
        self.gptq_related_blocks = trace_gptq_target_blocks(model) # get the transformer block list above
        #self.pre_transformer_layers = trace_embeddings_layers(model) # get the embeddings above
        self.dataloader = dataloader
        self.device = device
        self.is_ready = False
        # import pdb;pdb.set_trace()
        # args parameters can be passed via kwargs
        try:
            self.args = args
            self.dtype = next(iter(self.model.parameters())).dtype
            self.inp = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=self.dtype, device=self.device)
            self.cache = {'i': 0}
            # for opt, bloom, llama, etc, their inputs are different thus their cache structures vary
            # initialization
            # for special_input_terms in arch_inputs[self.args.arch]:
            #     self.cache[special_input_terms] = None
            self.out = torch.zeros_like(self.inp)
            self.is_ready = True
        except:
            pass
    
    @torch.no_grad()
    def pre_quantization(self):
        # by executing forward process, collect inputs of transformer blocks
        # process the embedding related layers, set devices
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer = embedding_layer.to(self.device)
        # hook inputs of transformer blocks
        # hook the first transformer block to obtain initial inputs

        # obtain the first layer inputs and registered to inputs
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].to(self.device)
        # layer_0_handle = self.gptq_related_blocks['transformers'][0].register_forward_hook(input_hook)
        self.gptq_related_blocks['transformers'][0] = InputHooker(self.gptq_related_blocks['transformers'][0], self.args.arch, self.inp, self.cache)
        for batch in self.dataloader:
            try:
                self.model(batch[0].to(self.device))
            except ValueError:
                pass

        # copy data from hookers
        # self.inp = self.gptq_related_blocks['transformers'][0].inp # no need to add this process
        # self.cache = self.gptq_related_blocks['transformers'][0].cache
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].module
        # t_layer = t_layer.cpu()
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].cpu()
        # after store inputs, locate embedding layers and transformer[0] back to cpu
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer.to(self.device)
        torch.cuda.empty_cache()
        print('Quantization prepared.')
    
    # def perform_transformer_forward(self, module, inp):
    #     assert self.is_ready, "calibration datasets are not prepared!"
    #     if self.args.arch == "bloom":
    #         return module(inp, attention_mask=self.cache['attention_mask'], alibi=self.cache['alibi'], )[0]
    #     elif self.args.arch == "opt":
    #         return module(inp, attention_mask=self.cache['attention_mask'], )[0]
    #     elif self.args.arch == 'llama':
    #         return module(inp, attention_mask=self.cache['attention_mask'], position_ids=self.cache['position_ids'])[0]
    #     else:
    #         raise NotImplementedError
    
    # def perform_transformer_forward_v2(self, module, inp, args, **kwargs):
    #     pass

    @torch.no_grad()
    def execute_quantization(self, means=None, stds=None):
        print("Begin ====>")
        self.pre_quantization()

        # quantizers = {}
        # Triggle GPTQ algorithm block by block.
        for block_idx in range(len(self.gptq_related_blocks['transformers'])):
            transformer_block = self.gptq_related_blocks['transformers'][block_idx].to(self.device)
            # trace all layers which can be quantized (Linear, Conv2d, etc.)
            sub_layers = find_layers(transformer_block)
            gptq_for_this_block = {}
            # import pdb;pdb.set_trace()
            for layer_name in sub_layers:
                gptq_for_this_block[layer_name] = GPTQ(sub_layers[layer_name])
                gptq_for_this_block[layer_name].quantizer = Quantizer()
                gptq_for_this_block[layer_name].quantizer.configure(self.args.wbits, perchannel=True, sym=self.args.sym, mse=False)

            def add_batch(_name):
                def tmp(_, inp, out):
                    gptq_for_this_block[_name].add_batch(inp[0].data, out.data)
                return tmp
            
            # register handles which add inputs and outputs to gptq object
            handles = []
            
            for layer_name in sub_layers:
                handles.append(sub_layers[layer_name].register_forward_hook(add_batch(layer_name)))
            # import pdb;pdb.set_trace()
            idx = self.cache.pop('i')
            for j in range(self.args.nsamples):
                # during the forward process, the batch data has been registered into gptq object, which contributes to quantization process.
                # self.out[j] = transformer_block(self.inp[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
                # method 1: use pre-defined methods
                # self.out[j] = self.perform_transformer_forward(transformer_block, self.inp[j].unsqueeze(0))
                # method 2: use dict passing
                self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
            self.cache['i'] = idx
            for h in handles:
                h.remove()
            
            # import pdb;pdb.set_trace()
            for layer_name in sub_layers:
                print(f"Quantizing layer {layer_name}")
                gptq_for_this_block[layer_name].fasterquant(percdamp=self.args.percdamp, groupsize=self.args.groupsize)
            # import pdb;pdb.set_trace()
            idx = self.cache.pop('i')
            for j in range(self.args.nsamples):
                # out should be quantized results
                # idx = self.cache.pop('i')
                # self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
                # self.cache[i] = idx
                # idx = self.cache.pop('i')
                self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
                # self.out[j] = self.perform_transformer_forward(transformer_block, self.inp[j].unsqueeze(0))
            self.cache['i'] = idx
            self.gptq_related_blocks['transformers'][block_idx] = transformer_block.cpu()
            del gptq_for_this_block
            torch.cuda.empty_cache()
            # iteratively replace the input with output (next block)
            self.inp, self.out = self.out, self.inp
        # import pdb;pdb.set_trace()
        self.model.config.use_cache = self.use_cache
    
    @torch.no_grad()
    def post_quantization(self, test_dataloader):
        print("Evaluation...")

        test_dataloader = test_dataloader.input_ids
        nsamples = test_dataloader.numel() // self.model.seqlen
        try:
            del self.inp
        except:
            pass
        self.inp = torch.zeros((nsamples, self.model.seqlen, self.model.config.hidden_size), dtype=self.dtype, device=self.device)

        try:
            del self.out
        except:
            pass
        self.out = torch.zeros_like(self.inp)
        
        use_cache = self.use_cache
        self.model.config.use_cache = False

        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer = embedding_layer.to(self.device)

        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].to(self.device)
        # layer_0_handle = self.gptq_related_blocks['transformers'][0].register_forward_hook(input_hook)
        self.gptq_related_blocks['transformers'][0] = InputHooker(self.gptq_related_blocks['transformers'][0], self.args.arch, self.inp, self.cache)
        for i in range(nsamples):
            batch = test_dataloader[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)].to(self.device)
            try:
                self.model(batch[0].to(self.device))
            except ValueError:
                pass

        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].module
        # t_layer = t_layer.cpu()
        self.gptq_related_blocks['transformers'][0] = self.gptq_related_blocks['transformers'][0].cpu()
        for embedding_name, embedding_layer in self.gptq_related_blocks["embeddings"].items():
            embedding_layer.to(self.device)
        torch.cuda.empty_cache()
        print('Evaluation dataset prepared. Begin evaluation...')

        # begin evaluation
        for block_idx in tqdm(range(len(self.gptq_related_blocks['transformers']))):
            transformer_block = self.gptq_related_blocks['transformers'][block_idx].to(self.device)
            # if args.nearest:
            #     subset = find_layers(layer)
            #     for name in subset:
            #         quantizer = Quantizer()
            #         quantizer.configure(
            #             args.wbits, perchannel=True, sym=args.sym, mse=False
            #         )
            #         W = subset[name].weight.data
            #         quantizer.find_params(W, weight=True)
            #         subset[name].weight.data = quantize(
            #             W, quantizer.scale, quantizer.zero, quantizer.maxq
            #         ).to(next(iter(layer.parameters())).dtype)

            idx = self.cache.pop('i')
            for j in range(nsamples):
                self.out[j] = transformer_block(self.inp[j].unsqueeze(0), **self.cache)[0]
            self.cache['i'] = idx
            self.gptq_related_blocks['transformers'][block_idx] = transformer_block.cpu()
            del transformer_block
            torch.cuda.empty_cache()
            self.inp, self.out = self.out, self.inp

        # to be modified
        self.model.transformer.ln_f = self.model.transformer.ln_f.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

        test_dataloader = test_dataloader.to(self.device)
        nlls = []
        for i in range(nsamples):
            hidden_states = self.inp[i].unsqueeze(0)
            hidden_states = self.model.transformer.ln_f(hidden_states)
            lm_logits = self.model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = test_dataloader[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))
        print(ppl.item())

#------------------------gptq source codes---------------------------------

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0] # output channels
        self.columns = W.shape[1] # input channels
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t()) # H = X*X, which should be a sysm matrix

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        # import pdb;pdb.set_trace()
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 # such channel makes no contribution to quantization computation

        # rearrange considering the diag's value
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp # add a average value of 
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count): # within a block, channel wise
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
