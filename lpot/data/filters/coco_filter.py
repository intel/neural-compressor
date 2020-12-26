from lpot.utils.utility import LazyImport
from .filter import Filter, filter_registry
tf = LazyImport('tensorflow')

@filter_registry(filter_type="LabelBalanceCOCORecord", framework="tensorflow")
class LabelBalanceCOCORecordFilter(Filter):
    def __init__(self, size=1):
        self.size = size

    def __call__(self, image, label):
        return tf.math.equal(len(label[0]), self.size)