import os
import gc
import shutil
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from .utils import torch
from torch.quantization import prepare, convert, get_default_qconfig
# from torch.ao.quantization import swap_module, get_default_custom_config_dict, get_default_static_quant_module_mappings
from packaging.version import Version
if Version(torch.__version__.split('+')[0]).release < Version('2.0.0').release:
    from torch.quantization.quantize import add_observer_
else:
    from torch.quantization.quantize import _add_observer_ as add_observer_
from accelerate.utils import set_module_tensor_to_device
from .utils import load_shell, get_named_children, update_module, load_tensor_from_shard, load_tensor


TMP_DIR = './layer_wise_quant_tmp_dir'
QCONFIG = [
    torch.ao.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
            ),
    torch.ao.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight=torch.quantization.PerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_symmetric, dtype=torch.qint8)
        ),
    torch.ao.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(reduce_range=False),
            weight=torch.quantization.default_per_channel_weight_observer),
    torch.ao.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(reduce_range=False),
            weight=torch.quantization.PerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_symmetric, dtype=torch.qint8)),
    torch.ao.quantization.get_default_qat_qconfig('fbgemm'),
    torch.ao.quantization.get_default_qat_qconfig('qnnpack'),
    torch.ao.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
            weight=torch.quantization.default_per_channel_weight_observer
    )
]


def mk_tmp_dir():
    os.makedirs(TMP_DIR, exist_ok=True)


def del_tmp_dir():
    shutil.rmtree(TMP_DIR)


class QDQLayer(torch.nn.Module):
    def __init__(self, module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.module = module
        self.dequant = torch.ao.quantization.DeQuantStub()
 
    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        X = self.dequant(X)
        return X


class NormalQuant:
    def __init__(self, model, qconfig=0, output_dir='saved_results', device='cpu'):
        model.to(device)
        self.q_model = model
        qconfig = 1
        if isinstance(qconfig, (int, float)):
            self.qconfig = QCONFIG[int(qconfig)]
        else:
            self.qconfig = qconfig
        self.modules = get_named_children(self.q_model)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def quantize(self, calib_data):
        for idx, (name, module) in enumerate(self.modules):
            if module.__class__.__name__ in ['Linear']:
                module = QDQLayer(module)
                self.modules[idx] = (name, module)
                update_module(self.q_model, name, module)
                module.qconfig = self.qconfig
        prepare(self.q_model, inplace=True)
        if isinstance(calib_data, torch.Tensor):
            self.q_model(calib_data)
        elif isinstance(calib_data, torch.utils.data.dataloader.DataLoader):
            pbar = tqdm(enumerate(calib_data), total=len(calib_data))
            try:
                for idx, input in pbar:
                    pbar.set_description(f'iter {idx}')
                    self.q_model(**input)
            except Exception:
                for idx, (input, label) in pbar:
                    self.q_model(**input)
        convert(self.q_model, inplace=True)
        torch.save(self.q_model, os.path.join(self.output_dir, 'pytorch_model.bin'))


class LayerWiseQuant:
    def __init__(self, pretrained_model_name_or_path, cls, output_dir='./saved_results', qconfig=0, device='cpu'):
        self.q_model = load_shell(pretrained_model_name_or_path, cls)
        self.fp32_model = deepcopy(self.q_model)
        self.path = pretrained_model_name_or_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if isinstance(qconfig, (int, float)):
            self.qconfig = QCONFIG[int(qconfig)]
        else:
            self.qconfig = qconfig
        self.modules = get_named_children(self.q_model)
        # self.qconfig = torch.quantization.float_qparams_weight_only_qconfig
        self.device = device
        self._handle = {}

    def quantize(self, calib_data):
        mk_tmp_dir()
        self._layer_wise_quantize(calib_data)
        self._save()
        del_tmp_dir()

    def _layer_wise_quantize(self, calib_data):
        for idx, (name, module) in enumerate(self.modules):
            if module.__class__.__name__ in ['Linear']:
                module = QDQLayer(module)
                self.modules[idx] = (name, module)
                update_module(self.q_model, name, module)
                module.qconfig = self.qconfig
        self._regist_hooks()

        self.q_model.eval()
        with torch.no_grad():
            if isinstance(calib_data, torch.Tensor):
                self.q_model(calib_data)
            elif isinstance(calib_data, torch.utils.data.dataloader.DataLoader):
                pbar = tqdm(enumerate(calib_data), total=len(calib_data))
                try:
                    for idx, input in pbar:
                        pbar.set_description(f'iter {idx}')
                        self.q_model(**input)
                except Exception:
                    for idx, (input, label) in pbar:
                        self.q_model(**input)
            else:
                self.q_model(**calib_data)
        self._remove_hooks()

    def _save(self, path=None):
        if path is None:
            path = self.output_dir
        for name, module in self.modules:
            self._load_state_dict(name)
            # if module.__class__.__name__ in ['Linear']:
            #     new_module = convert(module, inplace=False)
            #     torch.save(new_module, os.path.join(self.output_dir, f'{name}.pt'))
            #     del new_module
            # else:
            #     convert(module, inplace=True)
            new_module = convert(module, inplace=False)
            torch.save(new_module, os.path.join(self.output_dir, f'{name}.pt'))
            del new_module
            self._clean_weight(module, name)
        torch.save(self.fp32_model, os.path.join(self.output_dir, 'model_arch.pt'))

    def _regist_hooks(self):
        def forward_pre_hook(name):
            def load_value(param_name):
                if 'lm_head' in param_name and getattr(self.config, "tie_word_embeddings", True):
                    input_embeddings = self.q_model.get_input_embeddings()
                    for name, module in self.modules:
                        if module == input_embeddings:
                            param_name = name + '.' + param_name.split('.')[-1]
                prefix = self.q_model.base_model_prefix
                if 'pytorch_model.bin.index.json' in os.listdir(self.path):
                    value = load_tensor_from_shard(self.path, param_name, prefix)
                else:
                    value = load_tensor(os.path.join(self.path, 'pytorch_model.bin'), param_name, prefix)
                return value

            def hook(module, input):
                file_path = os.path.join(TMP_DIR, f'{name}.pt')
                if os.path.exists(file_path):
                    self._load_state_dict(name)
                else:
                    if isinstance(module, QDQLayer):
                        for n, _ in module.module.named_parameters():
                            value = load_value(name + '.' + n)
                            set_module_tensor_to_device(self.q_model, name + '.module.' + n, self.device, value)
                        prepare(module, inplace=True)
                    else:
                        for n, p in module.named_parameters():
                            param_name = name + '.' + n
                            value = load_value(param_name)
                            # from hf transformers.modeling_utils._load_state_dict_into_meta_model
                            set_module_tensor_to_device(self.q_model, param_name, self.device, value)
                # gc.collect()
                # from torch.ao.quantization.quantization_mappings import get_default_qconfig_propagation_list
                # add_observer_(module, get_default_qconfig_propagation_list())
            return hook
        
        def forward_hook(name):
            def hook(module, input, output):
                file_path = os.path.join(TMP_DIR, f'{name}.pt')
                if os.path.exists(TMP_DIR):
                    # torch.save(module.state_dict(), file_path)
                    torch.save(module.state_dict(), file_path)
                self._clean_weight(module, name)
            return hook

        for name, module in self.modules:
            self._handle[name] = [module.register_forward_pre_hook(forward_pre_hook(name))]
            self._handle[name] += [module.register_forward_hook(forward_hook(name))]

    def _remove_hooks(self):
        for handle in self._handle.values():
            [h.remove() for h in handle]

    def _clean_weight(self, module, name):
        if isinstance(module, QDQLayer):
            submodule = module.module
        else:
            submodule = module
        
        for n, m in submodule.named_parameters():
            is_buffer = n in submodule._buffers
            old_value = getattr(submodule, n)
            with torch.no_grad():
                if is_buffer:
                    submodule._buffers[n] = torch.zeros([0], device="meta")
                else:
                    param_cls = type(submodule._parameters[n])
                    kwargs = submodule._parameters[n].__dict__
                    new_value = torch.zeros([0], device="meta")
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to("meta")
                    submodule._parameters[n] = new_value
        if self.device == 'cpu':
            gc.collect()
        else:
            torch.cuda.empty_cache()


    def _load_state_dict(self,  module_name):
        file_path = os.path.join(TMP_DIR, f'{module_name}.pt')
        state_dict = torch.load(file_path)
        for n, p in state_dict.items():
            set_module_tensor_to_device(self.q_model, f'{module_name}.{n}', self.device, p)
