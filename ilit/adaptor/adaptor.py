from abc import abstractmethod

'''The framework backends supported by ilit, including tensorflow, mxnet and pytorch.

   User could add new backend support by implementing new Adaptor subclass under this directory.
   The naming convention of new Adaptor subclass should be something like ABCAdaptor, user
   could choose this framework backend by setting "abc" string in framework field of yaml.

   FRAMEWORKS variable is used to store all implemented Adaptor subclasses of framework backends.
'''
FRAMEWORKS = {}


def adaptor_registry(cls):
    '''The class decorator used to register all Adaptor subclasses.

       Args:
           cls (class): The class of register.
    '''
    assert cls.__name__.endswith(
        'Adaptor'), "The name of subclass of Adaptor should end with \'Adaptor\' substring."
    if cls.__name__[:-len('Adaptor')].lower() in FRAMEWORKS:
        raise ValueError('Cannot have two frameworks with the same name.')
    FRAMEWORKS[cls.__name__[:-len('Adaptor')].lower()] = cls
    return cls


class Adaptor(object):
    '''The base class of framework adaptor layer.

    '''

    def __init__(self, framework_specific_info):
        pass

    @abstractmethod
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        '''The function is used to do calibration and quanitization in post-training quantization.

           Args:
               tune_cfg(dict): The chosen tuning configuration.
               model (object): The model to do calibration.
               dataloader(object): The dataloader used to load calibration dataset.
               q_func (optional): training function for quantization aware training mode.
        '''
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model, dataloader, metric):
        '''The function is used to run evaluation on validation dataset.

           Args:
               model (object): The model to do calibration.
        '''
        raise NotImplementedError

    @abstractmethod
    def query_fw_capability(self, model):
        '''The function is used to return framework tuning capability.

           Args:
               model (object): The model to query quantization tuning capability.
        '''
        raise NotImplementedError

    @abstractmethod
    def query_fused_patterns(self, model):
        '''The function is used to run fused patterns in framework.

           Args:
               model (object): The model to do calibration.

           Return:
              [['conv', 'relu'], ['conv', 'relu', 'bn']]
        '''
        raise NotImplementedError

    @abstractmethod
    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[]):
        '''The function is used by tune strategy class for dumping tensor info.

           Args:
               model (object): The model to do calibration.
           Return:
               Numpy Array Dict
               {'op1': tensor, 'op2': tensor}
        '''
        raise NotImplementedError

    @abstractmethod
    def mapping(self, src_model, dst_model):
        '''The function is used to create a dict to map tensor name of src model to tensor name of dst model.

           Return:
               Dict
               {'src_op1': 'dst_op1'}
        '''
        raise NotImplementedError
