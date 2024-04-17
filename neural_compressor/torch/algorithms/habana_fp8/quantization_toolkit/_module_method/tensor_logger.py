from typing import List, Type, Tuple
import pandas as pd
from pathlib import Path
import torch.nn
from collections.abc import Iterable
from collections import defaultdict
import os
from .quant_util import get_named_modules, QuantizedModule
from .._quant_common.quant_config import Fp8cfg

config = Fp8cfg().cfg

# Tensor logger - Measures statistics from the network and dumps to use in quantization
# Finds all modules which are quantized module, and registers a forward hook for measurements
class TensorLogger:
    def __init__(self, module:torch.nn.Module, types:Tuple[Type[torch.nn.Module]]=(QuantizedModule)):
        self.named_modules = list(get_named_modules(module, types=types))
        self.data = defaultdict(list)
        self.handles = []
        for name, module in self.named_modules:
            module.name = name
            def forward_hook(module, input, output):
                if isinstance(input, torch.Tensor): input = (input,)
                for i, input_ in enumerate(input):
                    self.data[module.name].append(input_.abs().max())
            hook = module.register_forward_hook(hook=forward_hook)
            self.handles.append(hook)
        
    # Removes hooks after measurement is over
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    # dumps statistics for use in quantization
    def dump(self, path:str='stats.pt'):
        torch.save(self.data, path)
    
    # remove hooks + dump statistics
    def finish(self):
        print('Finished measuring, creating dump')
        path = config['dump_stats_path']
        excel_path = config['dump_stats_xlsx_path']
        for keys in self.data:
            self.data[keys] = [x.item() for x in self.data[keys]]
        self.remove_hooks()
        if os.path.exists(path):
            os.remove(path)
        self.dump(path)
        if excel_path != None:
            self.dump_xlsx(excel_path)

    # dumps statistics to xlsx for analysis
    def dump_xlsx(self, path = 'stats.xlsx'):
        df = pd.DataFrame.from_dict(self.data)
        # tmp_df = pd.DataFrame()
        # tmp_df['name'] = list(df.columns)
        # tmp_df['mean'] = df.mean().values
        # tmp_df['max'] = df.max().values
        # tmp_df['std'] = df.std().values
        df.to_excel(path)