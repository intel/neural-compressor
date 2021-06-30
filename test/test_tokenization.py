from lpot.experimental.data.transforms.tokenization import FullTokenizer
import unittest
import os
import shutil
from lpot.utils.utility import LazyImport
tf =  LazyImport('tensorflow') 

basic_text = ["un", "##aff", "##able"]
class TestFullTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs('val', exist_ok=True)
        vocab_file = 'val/temp.txt'
        with tf.io.gfile.GFile(vocab_file,"w+") as f:
            for vocab in basic_text:
                f.write(vocab + '\n')
        f.close()
    @classmethod
    def tearDownClass(cls):
        if os.path.exists('val'):
            shutil.rmtree('val')
    def test_tokenizer(self):
        tokenizer = FullTokenizer('val/temp.txt')
        ids = [2,1,0]
        tokens = basic_text[::-1]
        tokens_to_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(tokens_to_ids, ids)
        ids_to_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertEqual(ids_to_tokens, tokens)
        split_tokens = tokenizer.tokenize("unaffable")
        self.assertEqual(split_tokens, basic_text)
        split_tokens = tokenizer.tokenize("example")
        self.assertEqual(split_tokens, ["[UNK]"])


if __name__ == "__main__":
    unittest.main()
