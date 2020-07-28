"""Tests for tuner"""
import numpy as np
import unittest
import os
import yaml
import tensorflow as tf
import importlib
     
def build_fake_yaml():
    fake_yaml = '''
        framework: 
          - name: tensorflow
            inputs: x
            outputs: op_to_store
        device: cpu
        tuning:
          - strategy: fake
            accuracy_criterion:
              - relative: 0.01        
        snapshot:
          - path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml2():
    fake_yaml = '''
        framework: 
          - name: tensorflow
            inputs: x
            outputs: op_to_store
        device: cpu
        tuning:
          - strategy: test
            accuracy_criterion:
              - relative: 0.01
        snapshot:
          - path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filter=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
 
            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
        
        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
 
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])
        
        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')        
    return graph

def build_fake_strategy_1():
    with open(os.path.join(os.path.dirname(importlib.util.find_spec('ilit').origin), 'strategy/fake.py'), 'w', encoding='utf-8') as f:
        seq = [
            "import time\n",
            "from .strategy import strategy_registry, TuneStrategy\n",
            "@strategy_registry\n",
            "class FakeTuneStrategy(TuneStrategy):\n",
            "  def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):\n",
            "    super(FakeTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)\n",
            "  def __getstate__(self):\n",
            "    save_dict = super(FakeTuneStrategy, self).__getstate__()\n",
            "    save_dict['id'] = self.id\n",
            "    return save_dict\n",
            "  def traverse(self):\n",
            "    gen = (x for x in range(5))\n",
            "    while True:\n",
            "      self.id = next(gen)\n"
            "      if self.id == 2:\n",
            "        raise KeyboardInterrupt\n"
        ]
        f.writelines(seq)
    f.close()

def build_fake_strategy_2():
    with open(os.path.join(os.path.dirname(importlib.util.find_spec('ilit').origin), 'strategy/test.py'), 'w', encoding='utf-8') as f:
        seq = [
            "import time\n",
            "from .strategy import strategy_registry, TuneStrategy\n",
            "@strategy_registry\n",
            "class TestTuneStrategy(TuneStrategy):\n",
            "  def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):\n",
            "    super(TestTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)\n",
            "  def traverse(self):\n",
            "    assert self.id == 2\n"
        ]
        f.writelines(seq)
    f.close()

def build_dataloader():
    from ilit.data import DataLoader
    from ilit.data import DATASETS
    dataset = DATASETS('tensorflow')['dummy']
    dataloader = DataLoader('tensorflow', dataset)
    return dataloader

class TestTuner(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_strategy_1()
        build_fake_strategy_2()
        build_fake_yaml()
        build_fake_yaml2()
        self.dataloader = build_dataloader()

    @classmethod
    def tearDownClass(self):
        os.remove(os.path.join(os.path.dirname(importlib.util.find_spec('ilit').origin), 'strategy/test.py'))
        os.remove('fake_yaml.yaml')
        os.remove(os.path.join(os.path.dirname(importlib.util.find_spec('ilit').origin), 'strategy/fake.py'))
        os.remove('fake_yaml2.yaml')    
        os.rmdir('saved')

    def test_autosave(self):
        from ilit.strategy import strategy
        from ilit import tuner as iLit

        at = iLit.Tuner('fake_yaml.yaml')
        at.tune(
            self.constant_graph,
            q_dataloader=self.dataloader,
            eval_dataloader=self.dataloader
        )

    def test_resume(self):
        from ilit.strategy import strategy
        from ilit import tuner as iLit
        at = iLit.Tuner('fake_yaml2.yaml')
        snapshot_path = at.conf.usr_cfg.snapshot.path
        files = os.listdir(snapshot_path)
        record = 0
        for file in files:
            if file.endswith('.snapshot'):
                record += 1
                path = os.path.join(snapshot_path, file)
                at.tune(
                    self.constant_graph,
                    q_dataloader=self.dataloader,
                    eval_dataloader=self.dataloader,
                    resume_file = path
                    )
                os.remove(os.path.join(snapshot_path, file))
        self.assertGreater(record, 0)
        

if __name__ == "__main__":
    unittest.main()
