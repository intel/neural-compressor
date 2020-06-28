from .strategy import strategy_registry, TuneStrategy
from collections import OrderedDict
import itertools

@strategy_registry
class ExhaustiveTuneStrategy(TuneStrategy):
    '''The tuning strategy using exhaustive search in tuning space.

       Args:
           cfg (object): The configuration user specified.
           adaptor (object): The class object of framework adaptor.
           baseline (tuple): The baseline of fp32 model.
    '''
    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        super(ExhaustiveTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)

    def next_tune_cfg(self):
        # generate tuning space according to user chosen tuning strategy

        op_cfgs = {}
        op_cfgs['op'] = OrderedDict()
        for iterations in self.calib_iter:
            op_cfgs['calib_iteration'] = int(iterations)
            op_lists = []
            op_cfg_lists = []
            for op, configs in self.opwise_tune_cfgs.items():
                op_lists.append(op)
                op_cfg_lists.append(configs)
            for cfgs in itertools.product(*op_cfg_lists):
                index = 0
                for cfg in cfgs:
                    op_cfgs['op'][op_lists[index]] = cfg
                    index += 1

                yield op_cfgs

        return

