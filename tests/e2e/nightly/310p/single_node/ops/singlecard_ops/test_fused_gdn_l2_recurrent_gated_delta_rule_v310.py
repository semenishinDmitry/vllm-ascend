import pytest
import torch
import torch_npu

from vllm_ascend._310p.ops.fla.fused_gdn_l2_recurrent_gated_delta_rule import (
    npu_fused_gdn_l2_recurrent_gated_delta_rule_310,
)
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

torch_npu.npu.set_compile_mode(jit_compile=False)


def _golden(
    query,
    key,
    value,
    a_log,
    a,
    b,
    dt_bias,
    state,
    seq_lens,
    indices,
    accepted,
    scale,
    eps,
):
    q = query.float()
    k = key.float()
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + eps)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + eps)
    v = value.float()
    states = state.float().clone()
    alpha = torch.exp(-torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float()))
    beta = torch.sigmoid(b.float())
    output = torch.empty_like(v)
    nv = value.shape[1]
    nk = query.shape[1]
    seq_start = 0

    for batch_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor)
        initial_token = seq_start + int(accepted[batch_idx]) - 1
        initial_state = states[indices[initial_token]].clone()
        for head in range(nv):
            recurrent_state = initial_state[head]
            qk_head = head // (nv // nk)
            for token in range(seq_start, seq_start + seq_len):
                q_t = q[token, qk_head] * scale
                k_t = k[token, qk_head]
                recurrent_state = recurrent_state * alpha[token, head]
                residual = v[token, head] - (recurrent_state * k_t.unsqueeze(0)).sum(dim=-1)
                recurrent_state = recurrent_state + (
                    residual * beta[token, head]
                ).unsqueeze(-1) * k_t.unsqueeze(0)
                states[indices[token], head] = recurrent_state
                output[token, head] = (recurrent_state * q_t.unsqueeze(0)).sum(dim=-1)
        seq_start += seq_len

    return output.to(value.dtype), states.to(state.dtype)


@pytest.mark.skipif(not is_310p_hw(), reason="Requires an Ascend 310P device.")
@pytest.mark.parametrize("batch_size,mtp,nk,nv", [(1, 1, 4, 8), (2, 2, 4, 8)])
def test_fused_gdn_l2_recurrent_gated_delta_rule_v310(batch_size, mtp, nk, nv):
    enable_custom_op()
    torch.manual_seed(42)
    dk = 128
    dv = 128
    token_count = batch_size * mtp
    seq_lens = torch.full((batch_size,), mtp, dtype=torch.int32)
    indices = torch.arange(token_count, dtype=torch.int32)
    accepted = torch.full((batch_size,), mtp, dtype=torch.int32)
    query = torch.randn(token_count, nk, dk, dtype=torch.float16)
    key = torch.randn(token_count, nk, dk, dtype=torch.float16)
    value = torch.randn(token_count, nv, dv, dtype=torch.float16)
    a_log = torch.randn(nv, dtype=torch.float32)
    a = torch.randn(token_count, nv, dtype=torch.float16)
    b = torch.randn(token_count, nv, dtype=torch.float16)
    dt_bias = torch.randn(nv, dtype=torch.float32)
    state = torch.randn(token_count, nv, dv, dk, dtype=torch.float16)
    scale = dk**-0.5
    eps = 1e-6

    expected_output, expected_state = _golden(
        query,
        key,
        value,
        a_log,
        a,
        b,
        dt_bias,
        state,
        seq_lens,
        indices,
        accepted,
        scale,
        eps,
    )

    state_npu = state.npu()
    output_npu = npu_fused_gdn_l2_recurrent_gated_delta_rule_310(
        query.npu(),
        key.npu(),
        value.npu(),
        a_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        state_npu,
        seq_lens.npu(),
        indices.npu(),
        accepted.npu(),
        scale_value=scale,
        l2_epsilon=eps,
    )

    torch.testing.assert_close(output_npu.float().cpu(), expected_output.float(), rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(state_npu.float().cpu(), expected_state.float(), rtol=3e-3, atol=1e-2)
