from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and not f.startswith('__') and not f.endswith('__init__.py'):
        __import__(basename(f)[:-3], globals(), locals(), level=1)


from .dataset import DATASETS, Dataset, IterableDataset, dataset_registry

__all__ = [DATASETS, Dataset, IterableDataset, dataset_registry]
