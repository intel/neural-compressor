import copy
from collections import OrderedDict
import itertools
from .strategy import strategy_registry, TuneStrategy

@strategy_registry
class BasicTuneStrategy(TuneStrategy):
    '''The basic tuning strategy which tunes the low precision model with below order
       1. TODO: fuse
       2. model-wise tuning
       3. fallback tuning for all ops from bottom to top

       Args:
           cfg (object): The configuration user specified.
           adaptor (object): The class object of framework adaptor.
           baseline (tuple): The baseline of fp32 model.
    '''
    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        super(BasicTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)

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
                if acc >= best_acc:
                    best_acc = acc
                    best_cfg = copy.deepcopy(op_cfgs)

        if best_cfg == None:
            return

        ops_acc = OrderedDict()
        for op, configs in reversed(self.opwise_tune_cfgs.items()):
            op_cfgs = copy.deepcopy(best_cfg)
            for cfg in configs:
                if 'fp32' == cfg['activation']['dtype']:
                    op_cfgs['op'][op]['activation'].clear()
                    op_cfgs['op'][op]['activation']['dtype'] = 'fp32'
                    if 'weight' in cfg:
                        assert cfg['weight']['dtype'] == 'fp32'
                        op_cfgs['op'][op]['weight'].clear()
                        op_cfgs['op'][op]['weight']['dtype'] = 'fp32'
            yield op_cfgs
            acc, _ = self.last_tune_result
            if acc > best_acc:
                ops_acc[op] = acc

        op_cfgs = copy.deepcopy(best_cfg)
        if ops_acc != None:
            ordered_ops = sorted(ops_acc.keys(), key=lambda key:ops_acc[key], reverse=True)
            for op in ordered_ops:
                old_cfg = copy.deepcopy(op_cfgs['op'][op])
                op_cfgs['op'][op]['activation']['dtype'] = 'fp32'
                if 'weight' in op_cfgs['op'][op]:
                    op_cfgs['op'][op]['weight']['dtype'] = 'fp32'
                yield op_cfgs
                acc, _ = self.last_tune_result
                if acc <= best_acc:
                    op_cfgs['op'][op] = copy.deepcopy(old_cfg)
                else:
                    best_acc = acc

        return
