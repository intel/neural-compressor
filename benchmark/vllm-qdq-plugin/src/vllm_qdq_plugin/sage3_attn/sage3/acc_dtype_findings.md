# acc_dtype Investigation: FP16 PV Path Causes Black Output

## Summary

When using `--acc_dtype fp16_both_dot` with `sage3_refactored`, the generated
video frames are entirely black. Root cause: the **PV path** (not the QK path)
produces NaN/Inf when running in FP16.

## Background

The `ACC_DTYPE_MAP` in `standalone/sage3/api.py` maps an `acc_dtype` string to
three Triton dtype parameters:

```
(qk_dot_dtype, pv_dot_dtype, softmax_dtype)
```

For `fp16_both_dot` this is `(fp16, fp16, fp32)`.

These parameters control the kernel in `standalone/sage3/attention_kernel.py`:

| Parameter         | Controls (kernel lines)                                  |
|-------------------|----------------------------------------------------------|
| `qk_dot_dtype`    | Q/K operand cast (L83, L115), QK dot out_dtype (derived) |
| `pv_dot_dtype`    | P/V operand cast (L125, L184), PV dot out_dtype (derived), output accumulator (L88), rescaling (L169), accumulation (L191), normalization (L198) |
| `softmax_dtype`   | running_max, running_sum, softmax computation            |

The launcher derives `dot_out_dtype` from operand dtype:
```python
qk_dot_out_dtype = fp32 if qk_dot_dtype is bf16 else qk_dot_dtype
pv_dot_out_dtype = fp32 if pv_dot_dtype is bf16 else pv_dot_dtype
```

So with `fp16_both_dot`, both `tl.dot` calls also output in FP16.

## Isolation Test

Added `fp16_qk_only` = `(fp16, fp32, fp32)` to isolate the QK vs PV path.

| acc_dtype       | QK path | PV path | Result     |
|-----------------|---------|---------|------------|
| `fp32`          | fp32    | fp32    | OK         |
| `fp16_qk_only`  | fp16    | fp32    | OK         |
| `fp16_pv_only`  | fp32    | fp16    | **BLACK**  |
| `fp16_both_dot` | fp16    | fp16    | **BLACK**  |

**Conclusion: the PV path is the sole culprit.**

QK with FP16 is fine because D=128 dot products stay well within FP16 max
(~65504), and tensor cores accumulate internally in FP32 regardless of
`out_dtype`.

## Root Cause (confirmed)

**PV dot product overflow in FP16.** Confirmed by capturing real Q/K/V
tensors from CogVideoX-2b and running a tile-by-tile Python simulation
(see `standalone/reproduce_pv_fp16_black.py --replay --debug-kernel`).

The overflow happens at the very first tile (Q-tile=0, K-tile=0):

```
p_quantized max:     2688.0    (NVFP4 COMBINED_MAX = 448 × 6)
v_tile max:          3.75      (real V after quantization; random data ≈ 1)
theoretical PV max:  2688 × 3.75 × 128 = 1,290,240
FP16 max:            65,504
observed PV max:     63,904    (right at FP16 limit → some elements are Inf)
```

Random data doesn't trigger NaN because V values have std≈1. Real
CogVideoX-2b V tensors have **std=3.35, max=30.9** — the 2688× P-quant
scaling factor combined with large V values makes FP16 overflow inevitable.

The Inf propagates through the subsequent softmax normalization steps
(Inf × 0 → NaN, Inf - Inf → NaN), producing all-NaN output which the
image processor casts to all-zero (black frames).

### Note on accumulator underflow

The original hypothesis about FP16 accumulator underflow during online
softmax rescaling (`output *= alpha` where `alpha = exp(old_max - new_max)`)
is a secondary concern. The PV overflow happens first and is the primary
cause. However, even if PV overflow were fixed (e.g., by using fp32 for
`pv_dot_out_dtype`), the accumulator underflow would still degrade quality
for longer sequences.

## Reproduce

### Op-level reproducer (no model needed)

```bash
cd standalone

# Small shapes (N=512, 4 tiles) — accuracy degrades but no NaN
python reproduce_pv_fp16_black.py

# CogVideoX-2b shapes (N=1576, 13 tiles) — produces NaN
python reproduce_pv_fp16_black.py --real-shapes
```

Expected output with `--real-shapes`:
```
acc_dtype        | cos        | psnr       | max_abs      | has_nan  | has_inf  | all_zero
------------------------------------------------------------------------------------------
fp32             | 0.982252   | 32.328671  | 0.096680     | no       | no       | no
fp16_qk_only     | 0.982250   | 32.328301  | 0.096680     | no       | no       | no
fp16_pv_only     | nan        | nan        | nan          | YES      | no       | no
fp16_both_dot    | nan        | nan        | nan          | YES      | no       | no
```

Note: with small N (4 tiles), fp16_pv_only doesn't NaN but accuracy drops
sharply (cosine 0.98 → 0.40). More tiles = more rescaling steps = worse
underflow, eventually producing NaN.

### E2E reproducer (needs CogVideoX-2b model)

All commands run from `example/`:

```bash
cd example

# Baseline (works)
python cogvideox_infer.py --model cogvideox-2b \
    --attention_type sage3_refactored -i -n 1 -q --acc_dtype fp32

# QK-only FP16 (works)
python cogvideox_infer.py --model cogvideox-2b \
    --attention_type sage3_refactored -i -n 1 -q --acc_dtype fp16_qk_only

# PV-only FP16 (BLACK — isolates the problem)
python cogvideox_infer.py --model cogvideox-2b \
    --attention_type sage3_refactored -i -n 1 -q --acc_dtype fp16_pv_only

# Both FP16 (BLACK — original report)
python cogvideox_infer.py --model cogvideox-2b \
    --attention_type sage3_refactored -i -n 1 -q --acc_dtype fp16_both_dot
```

Output frames saved to `videos/cogvideox-2b/smoke/quick_e2e_sage3_refactored/`.

## Next Steps

The fix requires decoupling `pv_dot_dtype` into finer roles:

1. **PV dot operand dtype** — can be FP16 (this is where the speedup comes
   from: FP16 tensor cores).
2. **PV dot output dtype** — should be FP32 to avoid overflow from large
   P-quant values.
3. **Output accumulator dtype** — must be FP32 to survive rescaling by tiny
   alpha values.

This means `ACC_DTYPE_MAP` needs to expand from 3 to 5 parameters, or the
kernel should unconditionally use FP32 for the accumulator and dot outputs
while only using `pv_dot_dtype` for operand casting.
