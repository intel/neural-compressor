
from .adaptor import FRAMEWORKS
from .conf.config import Conf
from .policy import POLICIES
from .utils import logger
from .utils.utility import singleton
from .data import DATASETS, TRANSFORMS


@singleton
class Pruning(object):
    """This is base class of pruning object.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and tuning objectives (performance, memory footprint etc.).
       Tuner class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)

    def on_epoch_begin(self, epoch):
        for policy in self.policies:
            policy.on_epoch_begin(epoch)

    def on_batch_begin(self, batch_id):
        for policy in self.policies:
            policy.on_batch_begin(batch_id)

    def on_batch_end(self):
        for policy in self.policies:
            policy.on_batch_end()

    def on_epoch_end(self):
        for policy in self.policies:
            policy.on_epoch_end()
        stats, sparsity = self.adaptor.report_sparsity(self.model)
        logger.info(stats)
        logger.info(sparsity)

    def __call__(self, model, q_dataloader=None, q_func=None,
                 eval_dataloader=None, eval_func=None, resume_file=None):
        self.cfg = self.conf.usr_cfg

        framework_specific_info = {'device': self.cfg.device,
                                   'approach': self.cfg.quantization.approach,
                                   'random_seed': self.cfg.tuning.random_seed}
        if self.cfg.framework.name.lower() == 'tensorflow':
            framework_specific_info.update(
                {"inputs": self.cfg.framework.inputs, "outputs": self.cfg.framework.outputs})
        framework = self.cfg.framework.name.lower()
        self.adaptor = FRAMEWORKS[framework](framework_specific_info)

        self.model = model
        policies = {}
        for policy in POLICIES:
            for name in self.cfg["pruning"][policy]:
                policies[name] = {"policy_name": policy,
                                  "policy_spec": self.cfg["pruning"][policy][name]}
        self.policies = []
        for name, policy_spec in policies.items():
            print(policy_spec)
            self.policies.append(POLICIES[policy_spec["policy_name"]](
                self.model, policy_spec["policy_spec"], self.cfg, self.adaptor))
        return q_func(model)
