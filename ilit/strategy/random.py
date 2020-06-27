from .strategy import strategy_registry, TuneStrategy
import numpy as np

@strategy_registry
class RandomTuneStrategy(TuneStrategy):
    '''The tuning strategy using random search in tuning space.

       Args:
           cfg (object): The configuration user specified.
           adaptor (object): The class object of framework adaptor.
           baseline (tuple): The baseline of fp32 model.
    '''
    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        super(RandomTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)

    def next_tune_cfg(self):
        # generate tuning space according to user chosen tuning strategy

        np.random.seed(self.cfg.random_seed)
        op_cfgs = {}
        op_cfgs['calib_iteration'] = int(np.random.choice(self.calib_iter))
        op_cfgs['op'] = {}
        for op, configs in self.opwise_tune_cfgs.items():
            op_cfgs['op'][op] = np.random.choice(configs)

        yield op_cfgs

