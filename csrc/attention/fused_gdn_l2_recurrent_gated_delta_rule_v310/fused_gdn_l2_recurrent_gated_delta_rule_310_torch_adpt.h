/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_GDN_L2_RECURRENT_GATED_DELTA_RULE_V310_TORCH_ADPT_H
#define FUSED_GDN_L2_RECURRENT_GATED_DELTA_RULE_V310_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_fused_gdn_l2_recurrent_gated_delta_rule_310(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& a_log,
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& dt_bias, at::Tensor& state,
    const at::Tensor& actual_seq_lengths, const at::Tensor& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens, double scale_value, double softplus_beta, double threshold,
    double l2_epsilon) {
  at::Tensor output = at::empty(value.sizes(), value.options());
  EXEC_NPU_CMD(aclnnFusedGdnL2RecurrentGatedDeltaRuleV310, query, key, value, a_log, a, b, dt_bias, state,
               actual_seq_lengths, ssm_state_indices, num_accepted_tokens, static_cast<float>(scale_value),
               static_cast<float>(softplus_beta), static_cast<float>(threshold), static_cast<float>(l2_epsilon),
               output);
  return output;
}

}  // namespace vllm_ascend
#endif