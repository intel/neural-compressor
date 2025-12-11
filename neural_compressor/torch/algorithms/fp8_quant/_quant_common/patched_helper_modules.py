import os
import torch
import neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules as inc_modules
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
    PatchedVllmMixtureOfExpertsOpFP8 as INCPatchedVllmMixtureOfExpertsOpFP8,
    PatchedModuleFusedSDPA as INCPatchedModuleFusedSDPA,
)


class OoTPatchedVllmMixtureOfExpertsOpFP8(INCPatchedVllmMixtureOfExpertsOpFP8):
    def _slice_moe(
        self, x, expert_routing_table, router_weights, permuted_weights=True, activation="silu"
    ):
        batched_tokens = x.shape[0]
        kwargs = {}
        orig_mod = self.orig_mod

        if orig_mod.enable_moe_chunk:
            chunk_size = orig_mod.chunk_size_list[-1]
            for idx, threshold in enumerate(orig_mod.token_boundary_list):
                if batched_tokens <= threshold:
                    chunk_size = orig_mod.chunk_size_list[idx]
                    break
            kwargs = {
                "chunk_size": chunk_size,
                "total_experts": 256,
            }

        qinput = self.quant_input(x)
        extra_kwargs = kwargs
        experts_range = range(self.experts_used)
        w1_list = [self.w13_list[i].weight for i in experts_range]
        w2_list = [self.w2_list[i].weight for i in experts_range]
        scale_w1 = [self.w13_list[i].scale_weight for i in experts_range]
        scale_w2 = [self.w2_list[i].scale_weight for i in experts_range]

        def _inner_forward(
            cur_qinput, cur_expert_routing_table, cur_router_weights, scale_input, extra_kwargs
        ):
            output = self.dynamic_moe_op(
                hidden_states=cur_qinput,
                expert_routing_table=cur_expert_routing_table,
                router_weights=cur_router_weights,
                w12=w1_list,
                w3=w2_list,
                d_scale_w12=scale_w1,
                d_scale_w3=scale_w2,
                d_scale_hidden_states=scale_input,
                d_scale_intermediate_hidden_states=self.scale_intermediate,
                permuted_weights=False,
                activation=activation,
                experts_min=self.experts_min,
                experts_max=self.experts_max,
                **extra_kwargs,
            )
            return output

        if batched_tokens > orig_mod.moe_slice_length:
            final_hidden_states_list = []
            n_slice = (batched_tokens + orig_mod.moe_slice_length - 1) // orig_mod.moe_slice_length
            for i in range(n_slice):
                s = i * orig_mod.moe_slice_length
                e = batched_tokens if i == (n_slice - 1) else (i + 1) * orig_mod.moe_slice_length
                cur_qinput = qinput[s:e, ...]
                cur_expert_routing_table = expert_routing_table[s:e, ...]
                cur_router_weights = router_weights[s:e, ...]
                scale_input = self.scale_input
                cur_out = _inner_forward(
                    cur_qinput, cur_expert_routing_table, cur_router_weights, scale_input, extra_kwargs
                )
                final_hidden_states_list.append(cur_out)
            final_hidden_states = torch.cat(final_hidden_states_list, dim=0)
        else:
            final_hidden_states = _inner_forward(
                qinput, expert_routing_table, router_weights, self.scale_input, extra_kwargs
            )

        return final_hidden_states.view(-1, x.shape[1])

    def _forward_quant(self, *args, **kwargs):
        return super().forward_quant(*args, **kwargs)

    def forward_quant(
        self, hidden_states, expert_routing_table, router_weights, permuted_weights=True, activation="silu"
    ):
        enable_moe_slice = hasattr(self.orig_mod, "enable_moe_slice") and self.orig_mod.enable_moe_slice
        if not enable_moe_slice:
            return self._forward_quant(
                hidden_states, expert_routing_table, router_weights, permuted_weights, activation
            )
        else:
            return self._slice_moe(hidden_states, expert_routing_table, router_weights, permuted_weights, activation)


class OoTPatchedModuleFusedSDPA(INCPatchedModuleFusedSDPA):
    def __init__(self, mod, parent, mod_extra_config, *args, **kwargs):
        super().__init__(mod, parent, mod_extra_config, *args, **kwargs)
        self.qkv_slice_thld = int(os.getenv("PT_HPU_QKV_SLICE_SEQ_LEN_THLD", 4096))
        if self.qkv_slice_thld > 0:
            self.qkv_chunk_size = int(os.getenv("VLLM_FUSEDSDPA_QKV_SLICE_CHUNK_SIZE", self.qkv_slice_thld))

        impl_mapping = {
            'split_kv': self.fp8_fsdpa_split_kv,
            'slice_causal': self.fp8_fsdpa_slice_causal,
            'slice_qkv': self.fp8_fsdpa_slice_qkv,
        }
        qkv_slice_impl = os.getenv("PT_HPU_QKV_SLICE_IMPL", 'slice_qkv').lower()
        assert qkv_slice_impl in impl_mapping, (
            f"Invalid QKV slice implementation: {qkv_slice_impl}, "
            f"available options: {list(impl_mapping.keys())}"
        )

        self.fp8_fsdpa_impl = impl_mapping[qkv_slice_impl]

    def fp8_fsdpa_fwd(
        self,
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        scale,
        is_causal,
        softmax_mode,
    ):
        results = torch.ops.hpu.fp8_sdpa_recomp_fwd(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            scale,
            is_causal,
            True,  # requires_backward
            softmax_mode,  # softmax_mode
            self.scale_q,  # d_scale_q
            self.scale_k,  # d_scale_k
            self.scale_v,  # d_scale_v
            self.scale_amax,  # q_scale_s
            self.scale_output,  # q_scale_o
            self.descale_amax,  # d_scale_s
            False,  # is_amax_s
            False,  # is_amax_o
            None,  # valid_seq_len
            "right",  # seq_padding_type
            (-1, -1),  # window_size
            None,  # sink
        )
        return results

    def fp8_fsdpa_split_kv(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        assert attn_mask is not None, "Attention mask is required for FSDPA with prefix caching."
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)
        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import (
            is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        )
        gqa = is_gqa(q, k)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(q, k, v, attn_mask)

        # calculate the prefix SDPA w/o mask
        prefix_k = k[..., :prefix_len, :]
        prefix_v = v[..., :prefix_len, :]
        prefix_res = self.fp8_fsdpa_fwd(q, prefix_k, prefix_v, None, dropout_p, scale, False, softmax_mode)
        prefix_out, prefix_m, prefix_linv = (gqa_output_reshape(x) if gqa else x for x in prefix_res[:3])
        prefix_m = prefix_m.to(torch.float32)
        prefix_linv = prefix_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
        prefix_out = self.dequant_output(prefix_out).to(torch.float32)

        # calculate the causal part
        causal_k = k[..., prefix_len:, :]
        causal_v = v[..., prefix_len:, :]
        causal_mask = attn_mask[..., -q_len:]
        causal_res = self.fp8_fsdpa_fwd(q, causal_k, causal_v, causal_mask, dropout_p, scale, False, softmax_mode)
        causal_out, causal_m, causal_linv = (gqa_output_reshape(x) if gqa else x for x in causal_res[:3])
        causal_m = causal_m.to(torch.float32)
        causal_linv = causal_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
        causal_out = self.dequant_output(causal_out).to(torch.float32)

        new_m = torch.maximum(prefix_m, causal_m)
        prefix_linv_rescaled = (1.0 / prefix_linv) * torch.exp(prefix_m - new_m)
        causal_linv_rescaled = (1.0 / causal_linv) * torch.exp(causal_m - new_m)
        final_linv = 1.0 / (prefix_linv_rescaled + causal_linv_rescaled)
        final_out = (prefix_linv_rescaled * final_linv) * prefix_out + (
            causal_linv_rescaled * final_linv
        ) * causal_out

        return final_out

    def fp8_fsdpa_slice_causal(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        assert attn_mask is not None, "Attention mask is required for FSDPA with prefix caching."
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)
        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import (
            is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        )
        gqa = is_gqa(q, k)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(q, k, v, attn_mask)

        # calculate the prefix SDPA w/o mask
        prefix_k = k[..., :prefix_len, :]
        prefix_v = v[..., :prefix_len, :]
        prefix_res = self.fp8_fsdpa_fwd(q, prefix_k, prefix_v, None, dropout_p, scale, False, softmax_mode)
        prefix_out, prefix_m, prefix_linv = (gqa_output_reshape(x) if gqa else x for x in (prefix_res[:3]))
        prefix_m = prefix_m.to(torch.float32)
        prefix_linv = prefix_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
        prefix_out = self.dequant_output(prefix_out).to(torch.float32)

        # calculate the causal part
        chunk_outputs = []
        num_chunks = (q_len + self.qkv_chunk_size - 1) // self.qkv_chunk_size
        for q_chunk_idx in range(num_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.qkv_chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.qkv_chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :]

            last_out = prefix_out[..., q_start:q_end, :]
            last_m = prefix_m[..., q_start:q_end, :]
            last_linv = prefix_linv[..., q_start:q_end, :]

            for kv_chunk_idx in range(0, num_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.qkv_chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.qkv_chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :]
                v_chunk = v[..., kv_start:kv_end, :]

                is_causal_chunk = kv_chunk_idx == 0 and q_chunk_idx != 0
                is_causal_chunk = is_causal_chunk and q_chunk_size % 1024 == 0 and kv_chunk_size % 1024 == 0
                mask_chunk = (
                    attn_mask[..., q_start:q_end, kv_start:kv_end]
                    if kv_chunk_idx == 0 and not is_causal_chunk
                    else None
                )
                chunk_res = self.fp8_fsdpa_fwd(
                    q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p, scale, is_causal_chunk, softmax_mode
                )

                chunk_out, chunk_m, chunk_linv = (gqa_output_reshape(x) if gqa else x for x in (chunk_res[:3]))
                chunk_m = chunk_m.to(torch.float32)
                chunk_linv = chunk_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
                chunk_out = self.dequant_output(chunk_out).to(torch.float32)

                new_m = torch.maximum(last_m, chunk_m)
                last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                last_out = (last_linv_rescaled * last_linv) * last_out + (
                    chunk_linv_rescaled * last_linv
                ) * chunk_out
                last_m = new_m
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        return torch.cat(chunk_outputs, dim=-2)

    def fp8_fsdpa_slice_qkv(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        prefix_len = kv_len - q_len
        softmax_mode = softmax_mode if softmax_mode == "fp32" else "fast"
        assert attn_mask is not None, "Attention mask is required for FSDPA with prefix caching."
        if scale is None:
            scale = 1.0 / (q.shape[-1] ** 0.5)
        from habana_frameworks.torch.hpex.kernels.Fp8FusedSDPA import (
            is_gqa, gqa_input_reshape_fwd, gqa_output_reshape
        )
        gqa = is_gqa(q, k)
        if gqa:
            q, k, v, attn_mask = gqa_input_reshape_fwd(q, k, v, attn_mask)

        chunk_outputs = []
        num_q_chunks = (q_len + self.qkv_chunk_size - 1) // self.qkv_chunk_size
        num_prefix_chunks = (prefix_len + self.qkv_chunk_size - 1) // self.qkv_chunk_size
        for q_chunk_idx in range(num_q_chunks):
            q_start = q_len - (q_chunk_idx + 1) * self.qkv_chunk_size
            q_start = max(q_start, 0)
            q_end = q_len - q_chunk_idx * self.qkv_chunk_size
            q_chunk_size = q_end - q_start
            q_chunk = q[..., q_start:q_end, :]

            last_out = None
            last_m = None
            last_linv = None
            for kv_chunk_idx in range(num_prefix_chunks):
                kv_start = prefix_len - (kv_chunk_idx + 1) * self.qkv_chunk_size
                kv_start = max(kv_start, 0)
                kv_end = prefix_len - kv_chunk_idx * self.qkv_chunk_size
                k_chunk = k[..., kv_start:kv_end, :]
                v_chunk = v[..., kv_start:kv_end, :]

                chunk_res = self.fp8_fsdpa_fwd(
                    q_chunk, k_chunk, v_chunk, None, dropout_p, scale, False, softmax_mode
                )
                chunk_out, chunk_m, chunk_linv = (gqa_output_reshape(x) if gqa else x for x in chunk_res[:3])
                chunk_m = chunk_m.to(torch.float32)
                chunk_linv = chunk_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
                chunk_out = self.dequant_output(chunk_out).to(torch.float32)

                if last_out is None or last_m is None or last_linv is None:
                    last_out = chunk_out
                    last_m = chunk_m
                    last_linv = chunk_linv
                else:
                    new_m = torch.maximum(last_m, chunk_m)
                    last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                    chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                    last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                    last_out = (last_linv_rescaled * last_linv) * last_out + (
                        chunk_linv_rescaled * last_linv
                    ) * chunk_out
                    last_m = new_m

            for kv_chunk_idx in range(0, num_q_chunks - q_chunk_idx):
                kv_start = prefix_len + q_end - (kv_chunk_idx + 1) * self.qkv_chunk_size
                kv_start = max(kv_start, prefix_len)
                kv_end = prefix_len + q_end - kv_chunk_idx * self.qkv_chunk_size
                kv_chunk_size = kv_end - kv_start
                k_chunk = k[..., kv_start:kv_end, :]
                v_chunk = v[..., kv_start:kv_end, :]

                is_causal_chunk = kv_chunk_idx == 0 and q_chunk_idx != 0
                is_causal_chunk = is_causal_chunk and q_chunk_size % 1024 == 0 and kv_chunk_size % 1024 == 0
                mask_chunk = (
                    attn_mask[..., q_start:q_end, kv_start:kv_end]
                    if kv_chunk_idx == 0 and not is_causal_chunk
                    else None
                )
                chunk_res = self.fp8_fsdpa_fwd(
                    q_chunk, k_chunk, v_chunk, mask_chunk, dropout_p, scale, is_causal_chunk, softmax_mode
                )

                chunk_out, chunk_m, chunk_linv = (gqa_output_reshape(x) if gqa else x for x in chunk_res[:3])
                chunk_m = chunk_m.to(torch.float32)
                chunk_linv = chunk_linv.to(torch.float32) * (128.0 if softmax_mode != "fp32" else 1.0)
                chunk_out = self.dequant_output(chunk_out).to(torch.float32)

                if last_out is None or last_m is None or last_linv is None:
                    last_out = chunk_out
                    last_m = chunk_m
                    last_linv = chunk_linv
                else:
                    new_m = torch.maximum(last_m, chunk_m)
                    last_linv_rescaled = (1.0 / last_linv) * torch.exp(last_m - new_m)
                    chunk_linv_rescaled = (1.0 / chunk_linv) * torch.exp(chunk_m - new_m)
                    last_linv = 1.0 / (last_linv_rescaled + chunk_linv_rescaled)
                    last_out = (last_linv_rescaled * last_linv) * last_out + \
                        (chunk_linv_rescaled * last_linv) * chunk_out
                    last_m = new_m
            chunk_outputs.append(last_out)
        chunk_outputs = list(reversed(chunk_outputs))
        return torch.cat(chunk_outputs, dim=-2)

    def forward_quant(
        self,
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        softmax_mode="None",
        recompute=None,
        valid_seq_len=None,
        seq_padding_type="None",
    ):
        qinput = self.quant_q(q).detach()
        kinput = self.quant_k(k).detach()
        vinput = self.quant_v(v).detach()
        q_len = q.shape[-2]
        kv_len = k.shape[-2]

        # for prefill with prefix caching
        if (
            q.shape[0] == 1
            and self.qkv_slice_thld > 0
            and q_len != 1
            and q_len != kv_len
            and kv_len >= self.qkv_slice_thld
        ):
            output = self.fp8_fsdpa_impl(
                qinput, kinput, vinput, attn_mask, dropout_p, is_causal, scale, softmax_mode, valid_seq_len, seq_padding_type
            )
            return output.to(q.dtype)
        else:
            results = self.fp8_fused_sdpa(
                qinput,
                kinput,
                vinput,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                softmax_mode="None",
                d_scale_q=self.scale_q,
                d_scale_k=self.scale_k,
                d_scale_v=self.scale_v,
                q_scale_s=self.scale_amax,
                q_scale_o=self.scale_output,
                d_scale_s=self.descale_amax,
                is_amax_s=False,
                valid_seq_len=valid_seq_len,
                seq_padding_type=seq_padding_type,
            )
            output = results[0]
            d_out = self.dequant_output(output)
            return d_out


INC_APPLY_OOT_PATCH = os.environ.get("INC_APPLY_OOT_PATCH", "0").lower() in ("1", "true", "yes")
if INC_APPLY_OOT_PATCH:
    from neural_compressor.torch.utils import logger

    logger.info("=========================== Applying INC Out of Tree Patches ===========================")
    inc_modules.PatchedVllmMixtureOfExpertsOpFP8 = OoTPatchedVllmMixtureOfExpertsOpFP8
    inc_modules.PatchedModuleFusedSDPA = OoTPatchedModuleFusedSDPA
