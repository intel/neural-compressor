import unittest
import numpy as np
import sys
sys.path.insert(0, './')
from neural_compressor.config import HPOConfig
from neural_compressor.compression.hpo import (GridSearcher,
                                               DiscreteSearchSpace,
                                               ContinuousSearchSpace,
                                               SearchSpace,
                                               prepare_hpo,
                                               SimulatedAnnealingOptimizer)


class TestHPO(unittest.TestCase):
    search_space = {
        'learning_rate': SearchSpace((0.0001, 0.001)),
        'num_train_epochs': SearchSpace(bound=(20, 100), interval=1),
        'weight_decay': SearchSpace((0.0001, 0.001)),
        'cooldown_epochs': SearchSpace(bound=(0, 10), interval=1),
        'sparsity_warm_epochs': SearchSpace(bound=(0, 5), interval=1),
        'per_device_train_batch_size': SearchSpace((5, 20), 1)
    }

    def test_searcher(self):
        hpo_config = HPOConfig({'num_train_epochs': self.search_space['num_train_epochs'],
                                'cooldown_epochs': self.search_space['cooldown_epochs']}, searcher='grid')
        searcher = GridSearcher({'num_train_epochs': self.search_space['num_train_epochs'],
                                 'cooldown_epochs': self.search_space['cooldown_epochs']})
        conf_searcher = prepare_hpo(hpo_config)
        self.assertEqual(searcher.__class__, conf_searcher.__class__)
        for _ in range(5):
            self.assertEqual(searcher.suggest(), conf_searcher.suggest())
        hpo_config = HPOConfig(self.search_space, 'random')
        searcher = prepare_hpo(hpo_config)
        for _ in range(5):
            searcher.suggest()
        hpo_config = HPOConfig(self.search_space, 'bo')
        searcher = prepare_hpo(hpo_config)
        for _ in range(10):
            searcher.suggest()
            searcher.get_feedback(np.random.random())
        hpo_config = HPOConfig(self.search_space, 'xgb', higher_is_better=True, min_train_samples=3)
        searcher = prepare_hpo(hpo_config)
        for _ in range(5):
            searcher.suggest()
            searcher.get_feedback(np.random.random())
        for _ in range(5):
            param = searcher.suggest()
            searcher.feedback(param, np.random.random())

    def test_search_space(self):
        ds = DiscreteSearchSpace(bound=[0, 10])
        get_ds = SearchSpace(bound=[0, 10], interval=1)
        self.assertEqual(ds.__class__, get_ds.__class__)
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
