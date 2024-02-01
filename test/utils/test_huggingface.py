"""Tests for downloading int8 model from huggingface model hub."""

import shutil
import unittest

import torch
import transformers

from neural_compressor.model import Model
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream


class TestQuantization(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved_results", ignore_errors=True)

    def test_int8_huggingface_model(self):
        from neural_compressor.utils.load_huggingface import OptimizedModel

        model_name_or_path = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        model = OptimizedModel.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=None,
            cache_dir=None,
            revision=None,
            use_auth_token=None,
        )

        stat = model.state_dict()
        self.assertTrue(stat["classifier.module._packed_params.dtype"] == torch.qint8)

        from huggingface_hub import hf_hub_download

        resolved_weights_file = hf_hub_download(
            repo_id=model_name_or_path,
            filename="pytorch_model.bin",
        )
        q_config = torch.load(resolved_weights_file)["best_configure"]
        inc_model = Model(model)
        inc_model.q_config = q_config

        save_for_huggingface_upstream(inc_model, tokenizer, "saved_results")
        load_model = OptimizedModel.from_pretrained("saved_results")


if __name__ == "__main__":
    unittest.main()
