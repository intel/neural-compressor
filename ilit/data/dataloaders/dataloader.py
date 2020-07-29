from .base_dataloader import BaseDataLoader
from .tensorflow_dataloader import TensorflowDataLoader
from .mxnet_dataloader import  MXNetDataLoader
from .pytorch_dataloader import PyTorchDataLoader

DATALOADERS = {"tensorflow": TensorflowDataLoader,
               "mxnet": MXNetDataLoader,
               "pytorch":PyTorchDataLoader,}

class DataLoader(BaseDataLoader):
    """Entrance of all configured DataLoaders. Will dispatch the DataLoaders to framework specific one.
       Users will be not aware of the dispatching, and the Interface is unified.

    """
    def __init__(self, framework, dataset, batch_size=1, collate_fn=None, 
        last_batch='rollover', sampler=None, batch_sampler=None,
        num_workers=0, pin_memory=False):

        assert framework in ('tensorflow', 'pytorch', 'mxnet'), "framework support tensorflow pytorch mxnet"
        self.framework = framework
        super(DataLoader, self).__init__(dataset, batch_size=batch_size,
            collate_fn=collate_fn, last_batch=last_batch, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
        sampler, batch_sampler, num_workers, pin_memory):

        return DATALOADERS[self.framework](dataset, batch_size=batch_size,
            sampler=sampler, batch_sampler=batch_sampler, collate_fn=collate_fn,
            last_batch=last_batch,  num_workers=num_workers,
            pin_memory=pin_memory).dataloader

