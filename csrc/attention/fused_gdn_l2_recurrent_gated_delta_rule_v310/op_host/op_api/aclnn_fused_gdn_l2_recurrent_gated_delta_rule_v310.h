/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_FUSED_GDN_L2_RECURRENT_GATED_DELTA_RULE_V310_H
#define OP_API_ACLNN_FUSED_GDN_L2_RECURRENT_GATED_DELTA_RULE_V310_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate FusedGdnL2RecurrentGatedDeltaRuleV310 workspace
 * @param [in] query: float16
 * @param [in] key: float16
 * @param [in] value: float16
 * @param [in] aLog: float32
 * @param [in] a: float16
 * @param [in] b: float16
 * @param [in] dtBias: float32
 * @param [in] state: float16
 * @param [in] actualSeqLengths: int32
 * @param [in] ssmStateIndices: int32
 * @param [in] numAcceptedTokens: int32
 * @param [in] scaleValue: float32
 * @param [out] out: float16
 * @param [out] workspaceSize: workspace size
 * @param [out] executor: op executor
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnL2RecurrentGatedDeltaRuleV310GetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* aLog, const aclTensor* a,
    const aclTensor* b, const aclTensor* dtBias, aclTensor* stateRef, const aclTensor* actualSeqLengths,
    const aclTensor* ssmStateIndices, const aclTensor* numAcceptedTokens, float scaleValue, float softplusBeta,
    float threshold, float l2Epsilon, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief
 * @param [in] workspace: addr of workspace
 * @param [in] workspace_size: workspace size
 * @param [in] executor: op executor
 * @param [in] stream: acl stream
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnL2RecurrentGatedDeltaRuleV310(void* workspace,
                                                                                              uint64_t workspaceSize,
                                                                                              aclOpExecutor* executor,
                                                                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_FUSED_GDN_L2_RECURRENT_GATED_DELTA_RULE_V310_H
