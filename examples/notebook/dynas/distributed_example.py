from neural_compressor.conf.config import NASConfig
from neural_compressor.experimental.nas import NAS


if __name__ == '__main__':
    config = NASConfig(approach='dynas', search_algorithm='nsga2')

    config.dynas.supernet = 'ofa_mbv3_d234_e346_k357_w1.2'
    config.nas.search.seed = 61

    config.dynas.metrics = ['accuracy_top1', 'macs']

    config.dynas.population = 50
    config.dynas.num_evals = 250
    config.dynas.results_csv_path = 'results_ofa_dist.csv'
    config.dynas.batch_size = 128
    config.dynas.dataset_path = '/datasets/imagenet'
    config.dynas.distributed = True

    agent = NAS(config)
    results = agent.search()
