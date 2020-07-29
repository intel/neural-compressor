from ilit.utils.utility import LazyImport
from .base_dataloader import BaseDataLoader
torch = LazyImport('torch')

class PyTorchDataLoader(BaseDataLoader):
    """DataLoader for frameework PyTorch, we use the torch.utils.data.DataLoader

    """

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
        sampler, batch_sampler, num_workers, pin_memory):

        drop_last = False if last_batch == 'rollover' else True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            collate_fn=collate_fn, drop_last=drop_last, num_workers=num_workers,
            pin_memory=pin_memory, sampler=sampler, batch_sampler=batch_sampler)

