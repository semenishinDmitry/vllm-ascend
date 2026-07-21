import torch


def npu_fused_gdn_l2_recurrent_gated_delta_rule_310(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    scale_value: float = 1.0,
    softplus_beta: float = 1.0,
    threshold: float = 20.0,
    l2_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Run fused gating, Q/K L2 normalization, and recurrent delta rule."""
    return torch.ops._C_ascend.npu_fused_gdn_l2_recurrent_gated_delta_rule_310(
        query=query,
        key=key,
        value=value,
        a_log=a_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale_value=scale_value,
        softplus_beta=softplus_beta,
        threshold=threshold,
        l2_epsilon=l2_epsilon,
    )
