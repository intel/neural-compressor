"""Tests for downloading int8 model from huggingface model hub"""
import unittest
import torch


class TestQuantization(unittest.TestCase):

    def test_int8_huggingface_model(self):
        from neural_compressor.utils.load_huggingface import OptimizedModel

        model_name_or_path = 'Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static'

        model = OptimizedModel.from_pretrained(
                    model_name_or_path,
                    from_tf=bool(".ckpt" in model_name_or_path),
                    config=None,
                    cache_dir=None,
                    revision=None,
                    use_auth_token=None,
                )

        stat = model.state_dict()
        self.assertTrue(stat['classifier.module._packed_params.dtype'] == torch.qint8)


if __name__ == "__main__":
    unittest.main()
