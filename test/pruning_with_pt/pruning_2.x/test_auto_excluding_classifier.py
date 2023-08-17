import unittest

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

class NaiveMLP(nn.Module):
    def __init__(self, hidden_size=16):
        super(NaiveMLP, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ac2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 2, bias=True)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.ac1(x)
        x = self.linear2(x)
        x = self.ac2(x)
        x = self.linear3(x)
        return x

class TestPruning(unittest.TestCase):
    
    def test_pruning_basic(self):
        # import pdb;pdb.set_trace()
        hidden_size = 32
        model = NaiveMLP(hidden_size)
        # import classifier searching functions
        # A naive MLP model
        from neural_compressor.compression.pruner.model_slim.pattern_analyzer import ClassifierHeadSearcher
        searcher = ClassifierHeadSearcher(model)
        layer = searcher.search(return_name=True)
        assert layer == "linear3"
        del model

        # A Transformer model
        model_name_or_path = "textattack/distilbert-base-uncased-MRPC"
        task_name = "mrpc"
        num_labels = 2
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task=task_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
        searcher = ClassifierHeadSearcher(model)
        layer = searcher.search(return_name=True)
        assert layer == "classifier"

if __name__ == "__main__":
    unittest.main()
