import torch
import pytest
from neural_compressor.torch.experimental.fp4 import qdq_model
from neural_compressor.torch.experimental.fp4.modules import qdq_fp8


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)

class EmbeddingBagModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_bag = torch.nn.EmbeddingBag(10, 3, mode='sum')

    def forward(self, x, offsets=None):
        return self.embedding_bag(x, offsets=offsets)   


linear_input = torch.randn(2, 10)
embeddingbag_input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
embeddingbag_offsets = torch.tensor([0, 4], dtype=torch.long)

@pytest.mark.parametrize("x", [torch.randn(10, 3),])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("scale_format", ["pow2", "raw"])
@pytest.mark.parametrize("per_channel", [False, True])
def test_qdq_fp8(x, dtype, scale_format, per_channel):
    qdq_x, scale = qdq_fp8(x, dtype=dtype, scale_format=scale_format, per_channel=per_channel)
    print(f"Input x: {x}, dtype: {x.dtype}")
    print(f"qdq_x: {qdq_x}, dtype: {qdq_x.dtype}, scale_format: {scale_format}, per_channel: {per_channel}")
    print(f"scale: {scale}")

def test_linear_model():
    linear_model = LinearModel()
    linear_model.eval()
    with torch.no_grad():
        output = linear_model(linear_input)
        print(f"Linear model output: {output}")
    # Apply qdq_model to the linear_model
    linear_model = qdq_model(linear_model, dtype="nvfp4")
    print(f"Linear model: {linear_model}")
    with torch.no_grad():
        output = linear_model(linear_input)
        print(f"Linear model output after qdq: {output}")

def test_embeddingbag_model():
    embeddingbag_model = EmbeddingBagModel()
    embeddingbag_model.eval()
    with torch.no_grad():
        output = embeddingbag_model(embeddingbag_input, offsets=embeddingbag_offsets)
        print(f"EmbeddingBag model output: {output}")
    # Apply qdq_model to the embeddingbag_model
    embeddingbag_model = qdq_model(embeddingbag_model, dtype="nvfp4")
    print(f"EmbeddingBag model: {embeddingbag_model}")
    with torch.no_grad():
        output = embeddingbag_model(embeddingbag_input, offsets=embeddingbag_offsets)
        print(f"EmbeddingBag model output after qdq: {output}")
