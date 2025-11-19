import neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules as inc_modules
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import (
    PatchedVllmMixtureOfExpertsOpFP8 as INCPatchedVllmMixtureOfExpertsOpFP8,
)
import torch
import os


class OoTPatchedVllmMixtureOfExpertsOpFP8(INCPatchedVllmMixtureOfExpertsOpFP8):
    def _chunk_moe(self, x, expert_routing_table, router_weights, permuted_weights=True, activation="silu"):
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
        # tokens_num, hidden_dim = hidden_states.shape
        # extra_kwargs = self._get_extra_kwargs(tokens_num)
        extra_kwargs = kwargs
        experts_range = range(self.experts_used)
        w1_list = [self.w13_list[i].weight for i in experts_range]
        w2_list = [self.w2_list[i].weight for i in experts_range]
        scale_w1 = [self.w13_list[i].scale_weight for i in experts_range]
        scale_w2 = [self.w2_list[i].scale_weight for i in experts_range]

        def _inner_forward(cur_qinput, cur_expert_routing_table, cur_router_weights, scale_input, extra_kwargs):
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
        enable_moe_chunk = hasattr(self.orig_mod, "enable_moe_chunk") and self.orig_mod.enable_moe_chunk
        if not enable_moe_chunk:
            return self._forward_quant(
                hidden_states, expert_routing_table, router_weights, permuted_weights, activation
            )
        else:
            return self._chunk_moe(hidden_states, expert_routing_table, router_weights, permuted_weights, activation)


INC_APPLY_OOT_PATCH = os.environ.get("INC_APPLY_OOT_PATCH", "0").lower() in ("1", "true", "yes")
if INC_APPLY_OOT_PATCH:
    from neural_compressor.torch.utils import logger

    logger.info("=========================== Applying INC Out of Tree Patches ===========================")
    inc_modules.PatchedVllmMixtureOfExpertsOpFP8 = OoTPatchedVllmMixtureOfExpertsOpFP8
