from .patterns import BasePattern
import torch

REGS = {}


def register_reg(name):
    """Register a regularizator to the registry"""

    def register(reg):
        REGS[name] = reg
        return reg

    return register


def get_reg_type(config):
    for key in REGS.keys():  ##assume there is only one reg
        if config.get(key, None) != None:
            return key
    return None


def get_reg(config, modules, pattern):
    """Get registered regularizator class"""
    reg_type = config["reg_type"]
    if reg_type == None:
        return BaseReg(config, modules, pattern)
    if reg_type not in REGS.keys():
        assert False, f"regularizator does not support {reg_type}, currently only support {REGS.keys()}"
    return REGS[reg_type](config, modules, pattern, config["reg_coeff"])


class BaseReg:
    def __init__(self, config: dict, modules: dict, pattern: BasePattern):
        self.modules = modules
        self.config = config
        self.pattern = pattern

    def on_before_optimizer_step(self):
        pass

    def on_after_optimizer_step(self):
        pass


@register_reg("group_lasso")
class GroupLasso(BaseReg):
    def __init__(self, config: dict, modules: dict, pattern: BasePattern, coeff):
        super(GroupLasso, self).__init__(config, modules, pattern)
        assert "x" in self.config.pattern, "group lasso only supports NXM pattern"
        self.reg_terms = {}
        self.alpha = float(coeff)
        assert self.alpha >= 0, "group lasso only supports positive coeff"

    def on_before_optimizer_step(self):
        with torch.no_grad():
            if self.pattern.invalid_keys == None:
                self.pattern.check_layer_validity()
            for key in self.modules.keys():
                if key in self.pattern.invalid_keys:
                    continue
                grad = self.modules[key].weight.grad
                reg_term = self.pattern.reshape_orig_to_pattern(grad, key)
                reg_term = self.alpha / (torch.norm(reg_term, p=2, dim=[1, 3]) + 1e-12)
                reg_term[torch.isinf(reg_term)] = 0.0
                self.reg_terms[key] = reg_term

    def on_after_optimizer_step(self):  ##decoupled with grad decent
        with torch.no_grad():
            for key in self.modules.keys():
                if key in self.pattern.invalid_keys:
                    continue
                reg_term = self.pattern.reshape_reduced_to_orig(self.reg_terms[key], key,
                                                                self.modules[key].weight.shape)
                self.modules[key].weight -= reg_term
