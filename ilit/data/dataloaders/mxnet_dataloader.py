from ilit.utils.utility import LazyImport
from .base_dataloader import BaseDataLoader
mx = LazyImport('mxnet')

class MXNetDataLoader(BaseDataLoader):
    """DataLoader for frameework MXNet, we use the gluon.data.DataLoader
       (TODO) implement the DataIter dataloader while some models use it 

    """

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
        sampler, batch_sampler, num_workers, pin_memory):

        return mx.gluon.data.DataLoader(dataset, batch_size=batch_size,
            batchify_fn=collate_fn, last_batch=last_batch, num_workers=num_workers,
            pin_memory=pin_memory, sampler=sampler, batch_sampler=batch_sampler)
