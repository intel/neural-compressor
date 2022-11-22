from neural_compressor.experimental.data.dataloaders.fetcher import FETCHERS
from neural_compressor.experimental.data.dataloaders.sampler import BatchSampler
from neural_compressor.experimental.data.dataloaders.default_dataloader import DefaultDataLoader

# special dataloader for oob wide_deep model
class WidedeepDataloader(DefaultDataLoader):

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn, sampler,
                            batch_sampler, num_workers, pin_memory, shuffle, distributed):

        sampler = self._generate_sampler(dataset, distributed)
        self.batch_sampler = BatchSampler(sampler, batch_size, self.drop_last)
        self.fetcher = FETCHERS[self.dataset_type](dataset, collate_fn, self.drop_last, distributed)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                data[0][0] = data[0][0][0]
                yield data
            except StopIteration:
                return
                