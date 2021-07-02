#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import tensorflow as tf
from lpot.adaptor.tf_utils.util import get_estimator_graph

class TestEstimatorGraphConvert(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dst_path = '/tmp/.lpot/train.csv'
        self.titanic_file = tf.keras.utils.get_file(self.dst_path, \
            "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

    def test_get_estimator_graph(self):
        def train_input_fn():
          titanic = tf.data.experimental.make_csv_dataset(
              self.titanic_file, batch_size=32,
              label_name="survived")
          titanic_batches = (
              titanic.cache().repeat().shuffle(500)
              .prefetch(tf.data.experimental.AUTOTUNE))
          return titanic_batches
        age = tf.feature_column.numeric_column('age')
        cls = tf.feature_column.categorical_column_with_vocabulary_list('class', \
            ['First', 'Second', 'Third']) 
        embark = tf.feature_column.categorical_column_with_hash_bucket('embark_town', 32)
        import tempfile
        model_dir = tempfile.mkdtemp()
        model = tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=[embark, cls, age],
            n_classes=2
        )
        model = model.train(input_fn=train_input_fn, steps=100)
        result = model.evaluate(train_input_fn, steps=10)

        graph = get_estimator_graph(model, train_input_fn)

        self.assertTrue(isinstance(graph, tf.Graph)) 
        graph_def = graph.as_graph_def()
        self.assertGreater(len(graph_def.node), 1)


if __name__ == "__main__":
    unittest.main()
