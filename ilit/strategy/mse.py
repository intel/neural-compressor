from .strategy import strategy_registry, TuneStrategy
from collections import OrderedDict
import copy
import numpy as np

@strategy_registry
class MSETuneStrategy(TuneStrategy):
    '''The tuning strategy using MSE policy in tuning space.

       This MSE policy runs fp32 model and int8 model seperately to get all activation tensors,
       and then compares those tensors by MSE algorithm to order all ops with MSE distance for deciding
       the impact of each op to final accuracy. It will be used to define opwise tuning space by priority

       Args:
           cfg (object): The configuration user specified.
           adaptor (object): The class object of framework adaptor.
           baseline (tuple): The baseline of fp32 model.
    '''


    def __init__(self, model, cfg, dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        super(MSETuneStrategy, self).__init__(model, cfg, dataloader, q_func, eval_dataloader, eval_func, dicts)
        self.ordered_ops = None

    def __getstate__(self):
        save_dict = super(MSETuneStrategy, self).__getstate__()
        save_dict['ordered_ops'] = self.ordered_ops
        return save_dict

    def mse_metric_gap(self, fp32_tensor, dequantize_tensor):
        """
            caculate the euclidean distance between
            fp32 tensor and int8 dequantize tensor
        Args:
            fp32_tensr:
            dequantize_tensor:
        """
        fp32_max  = np.max(fp32_tensor)
        fp32_min  = np.min(fp32_tensor)
        dequantize_max = np.max(dequantize_tensor)
        dequantize_min = np.min(dequantize_tensor)
        fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
        dequantize_tensor = (dequantize_tensor - dequantize_min) / (dequantize_max - dequantize_min)
        diff_tensor = fp32_tensor - dequantize_tensor
        euclidean_dist = np.sum(diff_tensor ** 2)
        return euclidean_dist/fp32_tensor.size

    def next_tune_cfg(self):
        '''The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        '''

        # Model wise tuning
        op_cfgs = {}
        best_cfg = None
        best_acc = 0

        for iterations in self.calib_iter:
            op_cfgs['calib_iteration'] = int(iterations)
            for tune_cfg in self.modelwise_quant_cfgs:
                op_cfgs['op'] = OrderedDict()

                for op in self.opwise_quant_cfgs:
                    op_cfg = copy.deepcopy(self.opwise_quant_cfgs[op])
                    if len(op_cfg) > 0:
                        if tune_cfg not in op_cfg:
                            op_cfgs['op'][op] = copy.deepcopy(op_cfg[0])
                        else:
                            op_cfgs['op'][op] = copy.deepcopy(tune_cfg)
                    else:
                        op_cfgs['op'][op] = copy.deepcopy(self.opwise_tune_cfgs[op][0])

                yield op_cfgs
                acc, _ = self.last_tune_result
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = copy.deepcopy(op_cfgs)

        if best_cfg == None:
            return

        # Inspect FP32 and dequantized tensor
        if self.ordered_ops == None:
            op_lists = self.opwise_quant_cfgs.keys()
            fp32_tensor_dict = self.adaptor.inspect_tensor(self.model, self.calib_dataloader, op_lists, [1])
            best_qmodel = self.adaptor.quantize(best_cfg, self.model, self.calib_dataloader)
            dequantize_tensor_dict = self.adaptor.inspect_tensor(best_qmodel, self.calib_dataloader, op_lists, [1])

            ops_mse = {op:self.mse_metric_gap(fp32_tensor_dict[op], dequantize_tensor_dict[op]) for op in op_lists}
            self.ordered_ops = sorted(ops_mse.keys(),key=lambda key:ops_mse[key], reverse=True)

        op_cfgs = copy.deepcopy(best_cfg)
        if ops_mse != None:
            ordered_ops = sorted(ops_mse.keys(), key=lambda key:ops_mse[key], reverse=True)
            for op in ordered_ops:
                old_cfg = copy.deepcopy(op_cfgs['op'][op])
                op_cfgs['op'][op]['activation'] = {'dtype':'fp32'}
                if 'weight' in op_cfgs['op'][op].keys():
                    op_cfgs['op'][op]['weight'] = {'dtype':'fp32'}

                yield op_cfgs
                acc, _ = self.last_tune_result
                if acc <= best_acc:
                    op_cfgs['op'][op] = copy.deepcopy(old_cfg)
                else:
                    best_acc = acc

        return

