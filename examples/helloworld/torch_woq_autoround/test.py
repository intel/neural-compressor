from transformers import AutoModel, AutoTokenizer

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
from neural_compressor.adaptor.torch_utils.auto_round import get_dataloader

model_name = "EleutherAI/gpt-neo-125m"
float_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
dataloader = get_dataloader(tokenizer, seqlen=2048)

woq_conf = PostTrainingQuantConfig(
    approach="weight_only",
    op_type_dict={
        ".*": {  # match all ops
            "weight": {
                "dtype": "int",
                "bits": 4,
                "algorithm": "AUTOROUND",
            },
        }
    },
)
quantized_model = fit(model=float_model, conf=woq_conf, calib_dataloader=dataloader)
