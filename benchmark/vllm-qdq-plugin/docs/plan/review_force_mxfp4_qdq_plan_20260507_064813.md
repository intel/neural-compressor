## Goal

Review the current local changes for forced MXFP4 QDQ support and the move from ad-hoc `os.getenv` reads to a shared `envs` module, then fix any correctness issues.

## Finish Criteria

- Forced MXFP4 mode cannot silently fail because of env parsing or case handling.
- The package consistently reads public runtime switches through `vllm_qdq_plugin.envs`.
- The updated behavior is documented where users would look for it.
- Focused verification passes for import/syntax and env parsing behavior.

## Plan

1. Read the local diff and surrounding files to find correctness gaps.
2. Patch code paths that mishandle or incompletely apply the new env-based controls.
3. Add or update documentation for the new env knob if needed.
4. Run targeted verification and only stop once the new behavior is confirmed.
