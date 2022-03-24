import os
import argparse
import logging
import mxnet as mx

from rec2idx import IndexCreator

DATASET_URL = 'http://data.mxnet.io/data/val_256_q90.rec'


def main():
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Download imagenet dataset')
    parser.add_argument('--dataset_location', type=str, default='./data/',
                        help='Target directory for the dataset')
    args = parser.parse_args()
    dataset_loc = args.dataset_location

    logger.info('Downloading calibration dataset from %s to %s' % (DATASET_URL, dataset_loc))
    dataset_path = mx.test_utils.download(DATASET_URL, dirname=dataset_loc)
    idx_file_path = os.path.splitext(dataset_path)[0] + '.idx'
    if not os.path.isfile(idx_file_path):
        creator = IndexCreator(dataset_path, idx_file_path)
        creator.create_index()
        creator.close()


if __name__ == '__main__':
    main()
