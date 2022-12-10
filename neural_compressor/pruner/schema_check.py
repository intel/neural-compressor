from schema import Schema, And, Optional, Or
import yaml

try:
    from neural_compressor.conf.dotdict import DotDict
except:
    from .dot_dict import DotDict  ##TODO


def constructor_register(cls):
    yaml_key = "!{}".format(cls.__name__)

    def constructor(loader, node):
        instance = cls.__new__(cls)
        yield instance

        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    yaml.add_constructor(
        yaml_key,
        constructor,
        yaml.SafeLoader,
    )
    return cls


@constructor_register
class Pruner:
    """
    similiar to torch optimizer's interface
    """

    def __init__(self,
                 target_sparsity=None, pruning_type=None, pattern=None, op_names=None,
                 excluded_op_names=None,
                 start_step=None, end_step=None, pruning_scope=None, pruning_frequency=None,
                 min_sparsity_ratio_per_op=None, max_sparsity_ratio_per_op=None,
                 sparsity_decay_type=None, pruning_op_types=None, reg_type=None,
                 reduce_type=None, parameters=None, resume_from_pruned_checkpoint=None):
        self.pruner_config = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
            'reg_type': reg_type,
            'reduce_type': reduce_type,
            'parameters': parameters,
            'resume_from_pruned_checkpoint': resume_from_pruned_checkpoint
        })


# Schema library has different loading sequence priorities for different
# value types.
# To make sure the fields under dataloader.transform field of yaml file
# get loaded with written sequence, this workaround is used to convert
# None to {} in yaml load().
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:null', lambda loader, node: {})
# Add python tuple support because best_configure.yaml may contain tuple
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                lambda loader, node: tuple(loader.construct_sequence(node)))

weight_compression_schema = Schema({
    Optional('target_sparsity', default=0.9): float,
    Optional('pruning_type', default="snip_momentum"): str,
    Optional('pattern', default="4x1"): str,
    Optional('op_names', default=[]): list,
    Optional('excluded_op_names', default=[]): list,
    Optional('start_step', default=0): int,
    Optional('end_step', default=0): int,
    Optional('pruning_scope', default="global"): str,
    Optional('pruning_frequency', default=1): int,
    Optional('min_sparsity_ratio_per_op', default=0.0): float,
    Optional('max_sparsity_ratio_per_op', default=0.98): float,
    Optional('sparsity_decay_type', default="exp"): str,
    Optional('pruning_op_types', default=['Conv', 'Linear']): list,
    Optional('reg_type', default=None): str,
    Optional('reduce_type', default="mean"): str,
    Optional('parameters', default={"reg_coeff": 0.0}): dict,
    Optional('resume_from_pruned_checkpoint', default=False): str,
    Optional('pruners'): And(list, \
                             lambda s: all(isinstance(i, Pruner) for i in s))
})

approach_schema = Schema({
    Optional('weight_compression'): weight_compression_schema,
})

schema = Schema({
    Optional('model', default={'name': 'resnet50', \
                               'framework': 'NA', \
                               'inputs': [], 'outputs': []}): dict,

    Optional('version', default="1.0"): Or(str, float),

    Optional('pruning'): {
        Optional("approach"): approach_schema
    }
}

)
