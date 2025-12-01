import torch

E2M1_max = 6.0

E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

def cast_fp4(x):
    sign = torch.sign(x)
    sign_bit = (2 - sign) // 2
    ord_ = torch.sum(
        (x.abs().unsqueeze(-1) - E2M1_bounds.to(x.device)) > 0, dim=-1
    )
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
    return fp4_val

def fuse_uint4_to_uint8(x):
    # If the last dimension is odd, pad with zeros
    # If this behavior is not desired, please modify the code accordingly
    left_side = x[..., 0::2]  # Even indices (0, 2, 4...)
    right_side = x[..., 1::2]  # Odd indices (1, 3, 5...)
    new_data = right_side.clone() << 4  # Put odd indices (higher addresses) in high bits
    new_data[..., : left_side.shape[-1]] += left_side  # Put even indices in low bits
    return new_data
