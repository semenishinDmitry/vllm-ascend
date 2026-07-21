/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fused_gdn_l2_recurrent_gated_delta_rule_v310.cpp
 * \brief
 */
#include <dlfcn.h>
#include "aclnn_fused_gdn_l2_recurrent_gated_delta_rule_v310.h"
#include "../fused_gdn_l2_recurrent_gated_delta_rule_v310.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t QUERY_DIM_NUM = 3;
constexpr size_t KEY_DIM_NUM = 3;
constexpr size_t VALUE_DIM_NUM = 3;
constexpr size_t STATE_DIM_NUM = 4;

struct FusedGdnL2RecurrentGatedDeltaRuleV310Params {
  // mandatory
  const aclTensor* query{nullptr};
  const aclTensor* key{nullptr};
  const aclTensor* value{nullptr};
  const aclTensor* aLog{nullptr};
  const aclTensor* a{nullptr};
  const aclTensor* b{nullptr};
  const aclTensor* dtBias{nullptr};
  const aclTensor* state{nullptr};
  const aclTensor* actual_seq_lengths{nullptr};
  const aclTensor* ssm_state_indices{nullptr};
  const aclTensor* num_accepted_tokens{nullptr};
  // attrs
  float scaleValue{1.0f};
  float softplusBeta{1.0f};
  float threshold{20.0f};
  float l2Epsilon{1e-6f};
  // output
  const aclTensor* out{nullptr};
};

// support dtype
static const std::initializer_list<DataType> QKV_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> STATE_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> GATING_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<DataType> PARAM_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<DataType> SSM_TYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<DataType> ACC_TO_TYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<DataType> OUT_TYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};

static inline bool CheckNotNull(const FusedGdnL2RecurrentGatedDeltaRuleV310Params& params) {
  OP_CHECK_NULL(params.query, return false);
  OP_CHECK_NULL(params.key, return false);
  OP_CHECK_NULL(params.value, return false);
  OP_CHECK_NULL(params.state, return false);
  OP_CHECK_NULL(params.aLog, return false);
  OP_CHECK_NULL(params.a, return false);
  OP_CHECK_NULL(params.b, return false);
  OP_CHECK_NULL(params.dtBias, return false);
  OP_CHECK_NULL(params.actual_seq_lengths, return false);
  OP_CHECK_NULL(params.ssm_state_indices, return false);
  OP_CHECK_NULL(params.out, return false);

  return true;
}

static inline bool CheckDtypeVaild(const FusedGdnL2RecurrentGatedDeltaRuleV310Params& params) {
  OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.aLog, PARAM_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.a, GATING_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.b, GATING_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBias, PARAM_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, return false);

  if (params.num_accepted_tokens != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, return false);
  }

  OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
  return true;
}

static aclnnStatus CheckParams(FusedGdnL2RecurrentGatedDeltaRuleV310Params& params) {
  CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);
  OP_LOGD("FusedGdnL2RecurrentGatedDeltaRuleV310 check params success.");
  return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(FusedGdnL2RecurrentGatedDeltaRuleV310Params& params) {
  params.query->SetOriginalShape(params.query->GetViewShape());
  params.key->SetOriginalShape(params.key->GetViewShape());
  params.value->SetOriginalShape(params.value->GetViewShape());
  params.aLog->SetOriginalShape(params.aLog->GetViewShape());
  params.a->SetOriginalShape(params.a->GetViewShape());
  params.b->SetOriginalShape(params.b->GetViewShape());
  params.dtBias->SetOriginalShape(params.dtBias->GetViewShape());
  params.state->SetOriginalShape(params.state->GetViewShape());
  params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());
  params.ssm_state_indices->SetOriginalShape(params.ssm_state_indices->GetViewShape());

  return ACLNN_SUCCESS;
}
}  // namespace

aclnnStatus aclnnFusedGdnL2RecurrentGatedDeltaRuleV310GetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* aLog, const aclTensor* a,
    const aclTensor* b, const aclTensor* dtBias, aclTensor* stateRef, const aclTensor* actualSeqLengths,
    const aclTensor* ssmStateIndices, const aclTensor* numAcceptedTokens, float scaleValue, float softplusBeta,
    float threshold, float l2Epsilon, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnFusedGdnL2RecurrentGatedDeltaRuleV310,
                 DFX_IN(query, key, value, aLog, a, b, dtBias, stateRef, actualSeqLengths, ssmStateIndices,
                        numAcceptedTokens, scaleValue, softplusBeta, threshold, l2Epsilon),
                 DFX_OUT(out, stateRef));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  FusedGdnL2RecurrentGatedDeltaRuleV310Params params{query,
                                                     key,
                                                     value,
                                                     aLog,
                                                     a,
                                                     b,
                                                     dtBias,
                                                     stateRef,
                                                     actualSeqLengths,
                                                     ssmStateIndices,
                                                     numAcceptedTokens,
                                                     scaleValue,
                                                     softplusBeta,
                                                     threshold,
                                                     l2Epsilon,
                                                     out};

  CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  auto ret = PreProcess(params);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
  auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
  auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
  auto aLog_ = l0op::Contiguous(aLog, uniqueExecutor.get());
  auto a_ = l0op::Contiguous(a, uniqueExecutor.get());
  auto b_ = l0op::Contiguous(b, uniqueExecutor.get());
  auto dtBias_ = l0op::Contiguous(dtBias, uniqueExecutor.get());
  auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
  auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
  if (numAcceptedTokens != nullptr) {
    numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
  }

  auto out_ = l0op::Contiguous(out, uniqueExecutor.get());
  auto outRet = l0op::FusedGdnL2RecurrentGatedDeltaRuleV310(
      query_, key_, value_, aLog_, a_, b_, dtBias_, stateRef, actualSeqLengths_, ssmStateIndices_, numAcceptedTokens,
      scaleValue, softplusBeta, threshold, l2Epsilon, uniqueExecutor.get());
  if (outRet == nullptr) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
  if (ViewCopyResult == nullptr) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGdnL2RecurrentGatedDeltaRuleV310(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                       aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnFusedGdnL2RecurrentGatedDeltaRuleV310);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
