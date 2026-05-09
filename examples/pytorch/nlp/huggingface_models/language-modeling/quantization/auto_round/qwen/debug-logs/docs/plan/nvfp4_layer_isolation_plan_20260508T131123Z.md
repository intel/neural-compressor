# NVFP4 Layer Isolation Plan

## Question

Can the NVFP4-only long-context failure be isolated to a specific layer, and if so, how?

## Approach

- Confirm which parts of the checkpoint are actually quantized.
- Check whether the current repo already has a layerwise compare utility.
- Define a two-branch debug method:
  - model-layer isolation if the issue reproduces in direct forward
  - runtime-path isolation if the issue only reproduces in vLLM
- Save a findings note with a concrete step-by-step procedure.

## Finishing Criteria

- State clearly when “first bad layer” is a valid conclusion and when it is not.
- Provide a practical comparison method for BF16 vs NVFP4 on the same prompt.
- Include submodule-level hints based on the checkpoint’s quantization ignore list.
