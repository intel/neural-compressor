import torch

CRITERIAS = {}


def register_criteria(name):
    """Register a criteria to the registry"""

    def register(criteria):
        CRITERIAS[name] = criteria
        return criteria

    return register


def get_criteria(config, modules):
    """Get registered criteria class"""
    name = config["criteria_type"]
    if name not in CRITERIAS.keys():
        assert False, f"criterias does not support {name}, currently only support {CRITERIAS.keys()}"
    return CRITERIAS[name](modules, config)


class Criteria:
    def __init__(self, modules, config):
        self.scores = {}
        self.modules = modules
        self.config = config

    def on_step_begin(self):
        pass

    def on_after_optimizer_step(self):
        pass


@register_criteria('magnitude')
class MagnitudeCriteria(Criteria):
    def __init__(self, modules, config):
        super(MagnitudeCriteria, self).__init__(modules, config)

    def on_step_begin(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight.data
                self.scores[key] = p

@register_criteria('gradient')
class GradientCriteria(Criteria):
    def __init__(self, modules, config):
        super(GradientCriteria, self).__init__(modules, config)

    def on_after_optimizer_step(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p.grad)

@register_criteria('snip')
class SnipCriteria(Criteria):
    """
    please refer to SNIP: Single-shot Network Pruning based on Connection Sensitivity
    (https://arxiv.org/abs/1810.02340)
    """

    def __init__(self, modules, config):
        super(SnipCriteria, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"

    def on_after_optimizer_step(self):
        ##self.mask_weights()
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] = torch.abs(p * p.grad)


@register_criteria('snip_momentum')
class SnipMomentumCriteria(Criteria):
    def __init__(self, modules, config):
        super(SnipMomentumCriteria, self).__init__(modules, config)
        assert self.config.end_step > 0, "gradient based criteria does not work on step 0"
        for key in modules.keys():
            p = modules[key].weight
            self.scores[key] = torch.zeros(p.shape).to(p.device)

        self.alpha = 0.9
        self.beta = 1.0

    def on_after_optimizer_step(self):
        with torch.no_grad():
            for key in self.modules.keys():
                p = self.modules[key].weight
                self.scores[key] *= self.alpha
                self.scores[key] += self.beta * torch.abs(p * p.grad)


