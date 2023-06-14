import torch
import onnxruntime as ort
import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM

class LlamaLM(BaseLM):
    def __init__(
        self,
        device="cpu",
        pretrained="llama",
        tokenizer=None,
        batch_size=1,
        dtype: Optional[Union[str, torch.dtype]]="auto",
        user_model=None
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        device_list = set(
            ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        config = LlamaConfig.from_pretrained(pretrained)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions = ORTModelForCausalLM.load_model(
                        os.path.join(user_model, 'decoder_model.onnx'),
                        session_options=sess_options)

        self.model = ORTModelForCausalLM(sessions[0],
                                         config,
                                         user_model,
                                         use_cache=False
                                         )

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
        )

        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == "auto":
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps, torch.ones(inps.shape, dtype=torch.int64))[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)

