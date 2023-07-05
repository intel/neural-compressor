import unittest
import numpy as np
import sys
sys.path.insert(0, './')
from neural_compressor.compression.hpo import (GridSearcher,
                                               RandomSearcher,
                                               BayesianOptimizationSearcher,
                                               XgbSearcher,
                                               DiscreteSearchSpace,
                                               ContinuousSearchSpace,
                                               get_searchspace,
                                               SimulatedAnnealingOptimizer)


class TestHPO(unittest.TestCase):
    search_space = {
        'learning_rate': get_searchspace((0.0001, 0.001)),
        'num_train_epochs': get_searchspace(bound=(20, 100), interval=1),
        'weight_decay': get_searchspace((0.0001, 0.001)),
        'cooldown_epochs': get_searchspace(bound=(0, 10), interval=1),
        'sparsity_warm_epochs': get_searchspace(bound=(0, 5), interval=1),
        'per_device_train_batch_size': get_searchspace((5, 20), 1)
    }
    print(search_space)

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
            searcher.get_feedback(np.random.random())
        searcher = XgbSearcher(self.search_space, min_train_samples=3)
        for _ in range(5):
            searcher.suggest()
            searcher.get_feedback(np.random.random())
        for _ in range(5):
            param = searcher.suggest()
            searcher.feedback(param, np.random.random())

    def test_search_space(self):
        ds = DiscreteSearchSpace(bound=[0, 10])
        self.assertEqual(ds.index(1), ds.get_nth_value(1))
        ds = DiscreteSearchSpace(value=[1, 2, 3, 4])
        self.assertEqual(ds.get_all(), [1, 2, 3, 4])
        ds = DiscreteSearchSpace(bound=[0.01, 0.1])
        self.assertEqual(ds.interval, 0.01)
        self.assertIn(ds.get_value(), ds.get_all())
        self.assertEqual(ds.get_value(2), ds.get_nth_value(2))

        cs = ContinuousSearchSpace(bound=[0.01, 0.1])
        self.assertTrue(cs.get_value() >= 0.01)
        self.assertTrue(cs.get_value() < 0.1)
    
    def test_sa(self):
        def f(x):
            return np.mean(np.log(x**2), axis=1)
        points = np.random.randn(5, 6)
        optimizer = SimulatedAnnealingOptimizer(T0=100, Tf=0, alpha=0.9, higher_is_better=True)
        optimizer.gen_next_params(f, points)
        optimizer = SimulatedAnnealingOptimizer(T0=1, Tf=0.01, alpha=None, higher_is_better=False)
        optimizer.gen_next_params(f, points)


if __name__ == "__main__":
    unittest.main()
