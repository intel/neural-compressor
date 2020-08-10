from .dataset import dataset_registry, Dataset

@dataset_registry(dataset_type="bert", framework="pytorch", dataset_format='')
class BertDataset(Dataset):
    """Dataset used for model Bert.
       This Dataset is to construct from the Bert TensorDataset and not a full implementation
       from yaml cofig. The original repo link is: https://github.com/huggingface/transformers.
       When you want use this Dataset, you should add it before you initialize your DataLoader.
       (TODO) add end to end support for easy config by yaml by adding the method of
       load examples and process method.

    """
    def __init__(self, dataset, task, model_type='bert', transform=None):
        self.dataset = dataset
        assert task in ("classifier", "squad"), "Bert task support only classifier squad"
        self.task = task
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.task == 'classifier':
            inputs = {
                'input_ids':      sample[0],
                'attention_mask': sample[1],
                'labels':         sample[3]}

            if self.model_type != 'distilbert':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs['token_type_ids'] = sample[2] if self.model_type in [
                    'bert', 'xlnet'] else None
            sample = (inputs, inputs['labels'])

        elif self.task == 'squad':
            inputs = {
                'input_ids':       sample[0],
                'attention_mask':  sample[1],}
            if self.model_type != 'distilbert':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs['token_type_ids'] = sample[2] if self.model_type in [
                    'bert', 'xlnet'] else None
            if self.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': sample[4], 'p_mask': sample[5]})
            example_indices = sample[3]
            sample = (inputs, example_indices)
        return sample
