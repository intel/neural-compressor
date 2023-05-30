import unittest
import random
import sys
sys.path.insert(0, './')
from neural_compressor.compression.hpo import (GridSearcher,
                                               RandomSearcher,
                                               BayesianOptimizationSearcher, 
                                               XgbSearcher,
                                               DiscreteSearchSpace,
                                               ContinuousSearchSpace)


class TestHPO(unittest.TestCase):
    search_space = {
        'learning_rate': ContinuousSearchSpace((0.0001, 0.001)),
        'num_train_epochs': DiscreteSearchSpace(bound=(20, 100), interval=1),
        'weight_decay': ContinuousSearchSpace((0.0001, 0.001)),
        'cooldown_epochs': DiscreteSearchSpace(bound=(0, 10), interval=1),
        'sparsity_warm_epochs': DiscreteSearchSpace(bound=(0, 5), interval=1),
        'per_device_train_batch_size': DiscreteSearchSpace((5, 20), 1)
    }

    def test_searcher(self):
        searcher = GridSearcher({'num_train_epochs': self.search_space['num_train_epochs'],
                                 'cooldown_epochs': self.search_space['cooldown_epochs']})
        for _ in range(5):
            searcher.suggest()
        searcher = RandomSearcher(self.search_space)
        for _ in range(5):
            searcher.suggest()
        searcher = BayesianOptimizationSearcher(self.search_space)
        for _ in range(10):
            searcher.suggest()
            searcher.get_feedback(random.random())
        searcher = XgbSearcher(self.search_space, min_train_samples=3)
        for _ in range(5):
            searcher.suggest()
            searcher.get_feedback(random.random())
        for _ in range(5):
            param = searcher.suggest()
            searcher.feedback(param, random.random())


if __name__ == "__main__":
    unittest.main()
