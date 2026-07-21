/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file recurrent_gated_delta_rule_tiling_v310.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "fused_gdn_l2_recurrent_gated_delta_rule_v310_tiling.h"
#include "math_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/error_log.h"
#include <array>

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(FusedGdnL2RecurrentGatedDeltaRuleV310, FusedGdnL2RecurrentGatedDeltaRuleV310Tiling, 0);

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t A_LOG_INDEX = 3;
const size_t A_INDEX = 4;
const size_t B_INPUT_INDEX = 5;
const size_t DT_BIAS_INDEX = 6;
const size_t STATE_INDEX = 7;
const size_t CUSEQLENS_INDEX = 8;
const size_t SSM_STATE_INDICES_INDEX = 9;
const size_t ACC_TO_INDEX = 10;

const size_t QKV_DIM_NUM = 3;
const size_t PARAM_DIM_NUM = 1;
const size_t GATING_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t CUSEQLENS_DIM_NUM = 1;
const size_t SSM_STATE_INDICES_DIM_NUM = 1;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

const size_t MAX_MTP = 8;

void FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::InitCompileInfo() {
  auto platformInfoPtr = context_->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
    return;
  }
  const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
  compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

  if (compileInfo_.aivNum <= 0) {
    OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
    return;
  }
  tilingData_.vectorCoreNum = compileInfo_.aivNum;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; };

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetShapeAttrsInfo() {
  OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(AnalyzeDtype() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid dtypes."),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(AnalyzeShapes() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid shapes."),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(GetAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid attributes."),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(GetOptionalInput() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid GetOptionalInput."),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(AnalyzeFormat() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid Format."),
              return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::DoOpTiling() {
  OP_CHECK_IF(CalUbSize() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalUbSize failed."),
              return ge::GRAPH_FAILED);

  PrintTilingData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::DoLibApiTiling() {
  tilingKey_ = 0;
  return ge::GRAPH_SUCCESS;
};

uint64_t FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetTilingKey() const { return tilingKey_; };

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetWorkspaceSize() {
  // system workspace size is 16 * 1024 * 1024 = 16M;
  constexpr int64_t sysWorkspaceSize = 16777216;
  workspaceSize_ = sysWorkspaceSize;

  return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::PostTiling() {
  context_->SetBlockDim(tilingData_.vectorCoreNum);
  auto tilingDataSize = sizeof(FusedGdnL2RecurrentGatedDeltaRuleV310TilingData);
  errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                         reinterpret_cast<void*>(&tilingData_), tilingDataSize);
  if (ret != EOK) {
    OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
    return ge::GRAPH_FAILED;
  }
  context_->GetRawTilingData()->SetDataSize(tilingDataSize);

  size_t* workspaces = context_->GetWorkspaceSizes(1);  // set workspace
  OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is null"), return ge::GRAPH_FAILED);
  workspaces[0] = workspaceSize_;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckContext() {
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(QUERY_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(QUERY_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(KEY_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(KEY_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(VALUE_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(VALUE_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_LOG_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_LOG_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_INPUT_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(B_INPUT_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(DT_BIAS_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(DT_BIAS_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(STATE_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(STATE_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(CUSEQLENS_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(CUSEQLENS_INDEX));

  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(SSM_STATE_INDICES_INDEX));
  OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(SSM_STATE_INDICES_INDEX));

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::AnalyzeDtype() {
  auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
  auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
  auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
  OP_CHECK_IF(queryDtype != ge::DT_FLOAT16 || keyDtype != queryDtype || valueDtype != queryDtype,
              OP_LOGE(context_->GetNodeName(), "query/key/value dtype should be float16 and consistent"),
              return ge::GRAPH_FAILED);
  inputDtype_ = queryDtype;

  auto aDtype = context_->GetInputDesc(A_INDEX)->GetDataType();
  auto bDtype = context_->GetInputDesc(B_INPUT_INDEX)->GetDataType();
  auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
  OP_CHECK_IF(aDtype != queryDtype || bDtype != queryDtype || stateDtype != queryDtype,
              OP_LOGE(context_->GetNodeName(), "a/b/state dtype should match query dtype"), return ge::GRAPH_FAILED);

  auto aLogDtype = context_->GetInputDesc(A_LOG_INDEX)->GetDataType();
  auto dtBiasDtype = context_->GetInputDesc(DT_BIAS_INDEX)->GetDataType();
  OP_CHECK_IF(aLogDtype != ge::DT_FLOAT || dtBiasDtype != ge::DT_FLOAT,
              OP_LOGE(context_->GetNodeName(), "a_log and dt_bias dtype should be float32"), return ge::GRAPH_FAILED);

  auto cuSeqlensDtype = context_->GetInputDesc(CUSEQLENS_INDEX)->GetDataType();
  auto ssmStateIndicesDtype = context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetDataType();
  OP_CHECK_IF(cuSeqlensDtype != ge::DT_INT32 || ssmStateIndicesDtype != ge::DT_INT32,
              OP_LOGE(context_->GetNodeName(), "cuSeqlens dtype and ssmStateIndices dtype should be int32"),
              return ge::GRAPH_FAILED);

  if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
    auto numAcceptedTokensDtype = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetDataType();
    OP_CHECK_IF(numAcceptedTokensDtype != ge::DT_INT32,
                OP_LOGE(context_->GetNodeName(), "numAcceptedTokens dtype should be int32"), return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

bool FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckDimEqual(const gert::Shape a, const int64_t dimA, gert::Shape b,
                                                                const int64_t dimB, const std::string& nameA,
                                                                const std::string& nameB, const std::string& dimDesc) {
  if (a.GetDim(dimA) != b.GetDim(dimB)) {
    OP_LOGE(context_->GetNodeName(), "The %s of %s and %s should be the same, but %s is %ld while %s is %ld",
            dimDesc.c_str(), nameA.c_str(), nameB.c_str(), nameA.c_str(), a.GetDim(dimA), nameB.c_str(),
            b.GetDim(dimB));
    return false;
  }
  return true;
}

bool FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckDim(const gert::Shape shape, const size_t dim,
                                                           const std::string& dimDesc) {
  if (shape.GetDimNum() != dim) {
    OP_LOGE(context_->GetNodeName(), "The number of dimensions of %s should be %zu, but it is %zu", dimDesc.c_str(),
            dim, shape.GetDimNum());
    return false;
  }
  return true;
}

// Split shape checks/fill/scheduling decisions to improve readability and maintenance.
ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckShapeDimAndRelation(
    const gert::Shape& queryShape, const gert::Shape& keyShape, const gert::Shape& valueShape,
    const gert::Shape& aLogShape, const gert::Shape& aShape, const gert::Shape& bShape, const gert::Shape& dtBiasShape,
    const gert::Shape& stateShape, const gert::Shape& cuSeqlensShape, const gert::Shape& ssmStateShape) {
  if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(keyShape, QKV_DIM_NUM, "key") ||
      !CheckDim(valueShape, QKV_DIM_NUM, "value") || !CheckDim(aLogShape, PARAM_DIM_NUM, "a_log") ||
      !CheckDim(aShape, GATING_DIM_NUM, "a") || !CheckDim(bShape, GATING_DIM_NUM, "b") ||
      !CheckDim(dtBiasShape, PARAM_DIM_NUM, "dt_bias") || !CheckDim(stateShape, STATE_DIM_NUM, "state") ||
      !CheckDim(cuSeqlensShape, CUSEQLENS_DIM_NUM, "actual_seq_lengths") ||
      !CheckDim(ssmStateShape, SSM_STATE_INDICES_DIM_NUM, "ssm_state_indices")) {
    return ge::GRAPH_FAILED;
  }

  if (!CheckDimEqual(queryShape, DIM_0, keyShape, DIM_0, "query", "key", "T dimension") ||
      !CheckDimEqual(queryShape, DIM_1, keyShape, DIM_1, "query", "key", "Nk dimension") ||
      !CheckDimEqual(queryShape, DIM_2, keyShape, DIM_2, "query", "key", "Dk dimension") ||
      !CheckDimEqual(stateShape, DIM_1, valueShape, DIM_1, "state", "value", "Nv dimension") ||
      !CheckDimEqual(stateShape, DIM_2, valueShape, DIM_2, "state", "value", "Dv dimension") ||
      !CheckDimEqual(valueShape, DIM_0, queryShape, DIM_0, "value", "query", "T dimension") ||
      !CheckDimEqual(aShape, DIM_0, queryShape, DIM_0, "a", "query", "T dimension") ||
      !CheckDimEqual(aShape, DIM_1, valueShape, DIM_1, "a", "value", "Nv dimension") ||
      !CheckDimEqual(bShape, DIM_0, queryShape, DIM_0, "b", "query", "T dimension") ||
      !CheckDimEqual(bShape, DIM_1, valueShape, DIM_1, "b", "value", "Nv dimension") ||
      !CheckDimEqual(aLogShape, DIM_0, valueShape, DIM_1, "a_log", "value", "Nv dimension") ||
      !CheckDimEqual(dtBiasShape, DIM_0, valueShape, DIM_1, "dt_bias", "value", "Nv dimension") ||
      !CheckDimEqual(ssmStateShape, DIM_0, queryShape, DIM_0, "ssm_state_indices", "query", "T dimension") ||
      !CheckDimEqual(stateShape, DIM_3, queryShape, DIM_2, "state", "query", "Dk dimension")) {
    return ge::GRAPH_FAILED;
  }

  if (context_->GetOptionalInputShape(ACC_TO_INDEX) != nullptr) {
    const auto& acceptedShape = context_->GetOptionalInputShape(ACC_TO_INDEX)->GetOriginShape();
    if (!CheckDim(acceptedShape, CUSEQLENS_DIM_NUM, "num_accepted_tokens") ||
        !CheckDimEqual(acceptedShape, DIM_0, cuSeqlensShape, DIM_0, "num_accepted_tokens", "actual_seq_lengths",
                       "B dimension")) {
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

void FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::FillTilingShapeData(const gert::Shape& queryShape,
                                                                      const gert::Shape& valueShape,
                                                                      const gert::Shape& stateShape,
                                                                      const gert::Shape& cuSeqlensShape) {
  tilingData_.t = queryShape.GetDim(DIM_0);
  tilingData_.nk = queryShape.GetDim(DIM_1);
  tilingData_.dk = queryShape.GetDim(DIM_2);
  tilingData_.nv = valueShape.GetDim(DIM_1);
  tilingData_.dv = valueShape.GetDim(DIM_2);
  tilingData_.sBlockNum = stateShape.GetDim(DIM_0);
  tilingData_.b = cuSeqlensShape.GetDim(DIM_0);
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckShapeValueRangeAndRule() {
  OP_CHECK_IF(tilingData_.t == 0 || tilingData_.nk == 0 || tilingData_.nv == 0 || tilingData_.dk == 0 ||
                  tilingData_.dv == 0 || tilingData_.sBlockNum == 0 || tilingData_.b == 0,
              OP_LOGE(inputParams_.opName, "all operator dimensions must be positive"), return ge::GRAPH_FAILED);

  OP_CHECK_IF(tilingData_.nk > 256 || tilingData_.nv > 256 || tilingData_.dk > 512 || tilingData_.dv > 512,
              OP_LOGE(inputParams_.opName,
                      "nk and nv should no bigger than 256, dk and dv should no bigger than 512, but nk is %u, nv is "
                      "%u, dk is %u, dv is %u",
                      tilingData_.nk, tilingData_.nv, tilingData_.dk, tilingData_.dv),
              return ge::GRAPH_FAILED);

  OP_CHECK_IF(tilingData_.nv % tilingData_.nk != 0,
              OP_LOGE(inputParams_.opName, "nv should be an integer multiple of nk, but nv is %u, nk is %u",
                      tilingData_.nv, tilingData_.nk),
              return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

void FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::UpdateDynamicBlockDimByTaskUnits() {
  // Dynamic blockDim: do not launch more cores than effective (batch, head) task units.
  uint64_t taskUnits = static_cast<uint64_t>(tilingData_.b) * static_cast<uint64_t>(tilingData_.nv);
  if (taskUnits == 0) {
    taskUnits = 1;
  }
  uint64_t maxCoreNum = (compileInfo_.aivNum > 0) ? compileInfo_.aivNum : 1;
  uint64_t selectedCoreNum = (taskUnits < maxCoreNum) ? taskUnits : maxCoreNum;
  tilingData_.vectorCoreNum = static_cast<uint32_t>(selectedCoreNum);
  OP_LOGD(context_->GetNodeName(), "taskUnits: [%llu], selected vectorCoreNum: [%u]",
          static_cast<unsigned long long>(taskUnits), tilingData_.vectorCoreNum);
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCheckShapeDimAndRelation() {
  const auto& queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
  const auto& keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
  const auto& valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
  const auto& aLogShape = context_->GetInputShape(A_LOG_INDEX)->GetOriginShape();
  const auto& aShape = context_->GetInputShape(A_INDEX)->GetOriginShape();
  const auto& bShape = context_->GetInputShape(B_INPUT_INDEX)->GetOriginShape();
  const auto& dtBiasShape = context_->GetInputShape(DT_BIAS_INDEX)->GetOriginShape();
  const auto& stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
  const auto& cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
  const auto& ssmStateShape = context_->GetInputShape(SSM_STATE_INDICES_INDEX)->GetOriginShape();
  return CheckShapeDimAndRelation(queryShape, keyShape, valueShape, aLogShape, aShape, bShape, dtBiasShape, stateShape,
                                  cuSeqlensShape, ssmStateShape);
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleFillTilingShapeData() {
  const auto& queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
  const auto& valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
  const auto& stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
  const auto& cuSeqlensShape = context_->GetInputShape(CUSEQLENS_INDEX)->GetOriginShape();
  FillTilingShapeData(queryShape, valueShape, stateShape, cuSeqlensShape);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCheckShapeValueRangeAndRule() {
  return CheckShapeValueRangeAndRule();
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleUpdateDynamicBlockDimByTaskUnits() {
  UpdateDynamicBlockDimByTaskUnits();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleInitUbCalcContext() {
  ubCalcCtx_.ubSize = compileInfo_.ubSize;
  ubCalcCtx_.aNv = Ops::Transformer::CeilAlign(tilingData_.nv, static_cast<uint32_t>(16));  // 16 * 2 = 32B
  ubCalcCtx_.aDv = Ops::Transformer::CeilAlign(tilingData_.dv, static_cast<uint32_t>(16));  // 16 * 2 = 32B
  ubCalcCtx_.aDk = Ops::Transformer::CeilAlign(tilingData_.dk, static_cast<uint32_t>(16));  // 16 * 2 = 32B
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcFixedUbBytes() {
  ubCalcCtx_.fixedUbBytes = CalcFixedUbBytes(ubCalcCtx_.aNv, ubCalcCtx_.aDv, ubCalcCtx_.aDk);
  OP_CHECK_IF(ubCalcCtx_.fixedUbBytes >= static_cast<int64_t>(ubCalcCtx_.ubSize),
              OP_LOGE(context_->GetNodeName(), "fixed UB buffers exceed available UB"), return ge::GRAPH_FAILED);
  tilingData_.ubRestBytes = ubCalcCtx_.ubSize - ubCalcCtx_.fixedUbBytes;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcWorkingUbBytes() {
  ubCalcCtx_.workingUbBytes = CalcWorkingUbBytes(ubCalcCtx_.aNv, ubCalcCtx_.aDv, ubCalcCtx_.aDk);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcVStepCoeff() {
  ubCalcCtx_.coeff = CalcVStepCoeff(ubCalcCtx_.aDk, 1, 1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleFinalizeVStepFromUb() {
  return FinalizeVStepFromUb(ubCalcCtx_.ubSize, ubCalcCtx_.workingUbBytes, ubCalcCtx_.coeff);
}

// AnalyzeShapes now executes a deterministic rule-chain, easier to extend/maintain.
ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::AnalyzeShapes() {
  struct RuleItem {
    const char* name;
    HostRuleFn fn;
  };
  const std::array<RuleItem, 4> shapeRules = {{
      {"RuleCheckShapeDimAndRelation", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCheckShapeDimAndRelation},
      {"RuleFillTilingShapeData", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleFillTilingShapeData},
      {"RuleCheckShapeValueRangeAndRule",
       &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCheckShapeValueRangeAndRule},
      {"RuleUpdateDynamicBlockDimByTaskUnits",
       &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleUpdateDynamicBlockDimByTaskUnits},
  }};
  for (const auto& rule : shapeRules) {
    OP_CHECK_IF((this->*(rule.fn))() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "AnalyzeShapes rule failed: %s", rule.name), return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

bool FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CheckFormat(ge::Format format, const std::string& Desc) {
  if (format == ge::FORMAT_FRACTAL_NZ) {
    OP_LOGE(context_->GetNodeName(), "%s format not support NZ", Desc.c_str());
    return false;
  }
  return true;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::AnalyzeFormat() {
  if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX)->GetStorageFormat(), "query") ||
      !CheckFormat(context_->GetInputDesc(KEY_INDEX)->GetStorageFormat(), "key") ||
      !CheckFormat(context_->GetInputDesc(VALUE_INDEX)->GetStorageFormat(), "value") ||
      !CheckFormat(context_->GetInputDesc(A_LOG_INDEX)->GetStorageFormat(), "a_log") ||
      !CheckFormat(context_->GetInputDesc(A_INDEX)->GetStorageFormat(), "a") ||
      !CheckFormat(context_->GetInputDesc(B_INPUT_INDEX)->GetStorageFormat(), "b") ||
      !CheckFormat(context_->GetInputDesc(DT_BIAS_INDEX)->GetStorageFormat(), "dt_bias") ||
      !CheckFormat(context_->GetInputDesc(STATE_INDEX)->GetStorageFormat(), "state") ||
      !CheckFormat(context_->GetInputDesc(CUSEQLENS_INDEX)->GetStorageFormat(), "actual_seq_lengths") ||
      !CheckFormat(context_->GetInputDesc(SSM_STATE_INDICES_INDEX)->GetStorageFormat(), "ssm_state_indices")) {
    return ge::GRAPH_FAILED;
  }

  if (context_->GetOptionalInputDesc(ACC_TO_INDEX) != nullptr) {
    auto numAcceptedTokensFormat = context_->GetOptionalInputDesc(ACC_TO_INDEX)->GetStorageFormat();
    OP_CHECK_IF(numAcceptedTokensFormat == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context_->GetNodeName(), "numAcceptedTokens format not support NZ"), return ge::GRAPH_FAILED);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetAttrs() {
  auto attrs = context_->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
  const float* scale = attrs->GetAttrPointer<float>(0);
  const float* softplusBeta = attrs->GetAttrPointer<float>(1);
  const float* threshold = attrs->GetAttrPointer<float>(2);
  const float* l2Epsilon = attrs->GetAttrPointer<float>(3);
  OP_CHECK_IF(scale == nullptr || softplusBeta == nullptr || threshold == nullptr || l2Epsilon == nullptr,
              OP_LOGE(context_->GetNodeName(), "attribute pointer is null"), return ge::GRAPH_FAILED);
  OP_CHECK_IF(*softplusBeta <= 0.0f || *l2Epsilon < 0.0f,
              OP_LOGE(context_->GetNodeName(), "softplus_beta must be positive and l2_epsilon non-negative"),
              return ge::GRAPH_FAILED);
  tilingData_.scale = *scale;
  tilingData_.softplusBeta = *softplusBeta;
  tilingData_.threshold = *threshold;
  tilingData_.l2Epsilon = *l2Epsilon;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::GetOptionalInput() {
  if (context_->GetOptionalInputDesc(ACC_TO_INDEX) == nullptr) {
    tilingData_.hasAcceptedTokens = 0;
  } else {
    tilingData_.hasAcceptedTokens = 1;
  }

  return ge::GRAPH_SUCCESS;
}

void FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::PrintTilingData() {
  OP_LOGD(context_->GetNodeName(), "vectorCoreNum: [%u]", tilingData_.vectorCoreNum);
  OP_LOGD(context_->GetNodeName(), "ubCalSize: [%u]", tilingData_.ubCalSize);
  OP_LOGD(context_->GetNodeName(), "ubRestBytes: [%u]", tilingData_.ubRestBytes);
  OP_LOGD(context_->GetNodeName(), "t: [%u]", tilingData_.t);
  OP_LOGD(context_->GetNodeName(), "nk: [%u]", tilingData_.nk);
  OP_LOGD(context_->GetNodeName(), "dk: [%u]", tilingData_.dk);
  OP_LOGD(context_->GetNodeName(), "nv: [%u]", tilingData_.nv);
  OP_LOGD(context_->GetNodeName(), "dv: [%u]", tilingData_.dv);
  OP_LOGD(context_->GetNodeName(), "sBlockNum: [%u]", tilingData_.sBlockNum);
  OP_LOGD(context_->GetNodeName(), "b: [%u]", tilingData_.b);
  OP_LOGD(context_->GetNodeName(), "vStep: [%u]", tilingData_.vStep);
  OP_LOGD(context_->GetNodeName(), "stateOutBufferNum: [%u]", tilingData_.stateOutBufferNum);
  OP_LOGD(context_->GetNodeName(), "attnOutBufferNum: [%u]", tilingData_.attnOutBufferNum);
  OP_LOGD(context_->GetNodeName(), "scale: [%f]", tilingData_.scale);
  OP_LOGD(context_->GetNodeName(), "softplusBeta: [%f]", tilingData_.softplusBeta);
  OP_LOGD(context_->GetNodeName(), "threshold: [%f]", tilingData_.threshold);
  OP_LOGD(context_->GetNodeName(), "l2Epsilon: [%f]", tilingData_.l2Epsilon);
  OP_LOGD(context_->GetNodeName(), "hasAcceptedTokens: [%u]", tilingData_.hasAcceptedTokens);
}

int64_t FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CalcFixedUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const {
  int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv);  // 4 for qInQueue_ & kInQueue_, 2 for vInQueue_
  usedUbBytes += 128;                                   // reserve 128 Bytes
  // a/b fp16, generated gate/beta fp32, gating mask, fp32 a_log/dt_bias and two scratch vectors.
  usedUbBytes += MAX_MTP * 13 * aNv;
  usedUbBytes += 16 * aNv;
  return usedUbBytes;
}

int64_t FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CalcWorkingUbBytes(int64_t aNv, int64_t aDv, int64_t aDk) const {
  int64_t usedUbBytes = CalcFixedUbBytes(aNv, aDv, aDk);
  usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv);  // fp32 q/k/v working tensors
  usedUbBytes += 256 + 128;  // bank conflict padding (256B kInUb↔stateInUb, 128B stateInUb↔broadTmpInUb)
  return usedUbBytes;
}

int64_t FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CalcVStepCoeff(int64_t aDk, uint32_t stateOutBufferNum,
                                                                    uint32_t attnOutBufferNum) const {
  int64_t coeff = (2 + static_cast<int64_t>(2 * stateOutBufferNum)) * aDk +
                  static_cast<int64_t>(4 * attnOutBufferNum);  // stateIn/stateOut/attnOut queues
  coeff += (4 + 4 + 2) * aDk + 4 + 4;                          // stateInUb/broadTmpInUb/foldTmpUb/deltaInUb/attnInUb
  return coeff;
}

bool FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::EvaluateBufferProfile(int64_t ubSize, int64_t usedUbBytes,
                                                                        int64_t aDk, uint32_t stateOutBufferNum,
                                                                        uint32_t attnOutBufferNum,
                                                                        BufferProfile& profile) const {
  int64_t coeff = CalcVStepCoeff(aDk, stateOutBufferNum, attnOutBufferNum);
  int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;  // 8 * sizeof(float) = 32
  if (vStep < 8) {
    return false;
  }
  int64_t repeatTime = Ops::Transformer::CeilDiv(tilingData_.dv, static_cast<uint32_t>(vStep));
  vStep = Ops::Transformer::CeilAlign(Ops::Transformer::CeilDiv(tilingData_.dv, static_cast<uint32_t>(repeatTime)),
                                      static_cast<uint32_t>(8));
  if (vStep < 8) {
    return false;
  }
  profile.stateOutBufferNum = stateOutBufferNum;
  profile.attnOutBufferNum = attnOutBufferNum;
  profile.vStep = static_cast<uint32_t>(vStep);
  profile.repeatTime = static_cast<uint32_t>(repeatTime);
  profile.valid = true;
  return true;
}

bool FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::IsBetterProfile(const BufferProfile& candidate,
                                                                  const BufferProfile& current) const {
  if (!current.valid) {
    return true;
  }
  if (candidate.repeatTime != current.repeatTime) {
    return candidate.repeatTime < current.repeatTime;
  }
  uint32_t candidateDepth = candidate.stateOutBufferNum + candidate.attnOutBufferNum;
  uint32_t currentDepth = current.stateOutBufferNum + current.attnOutBufferNum;
  if (candidateDepth != currentDepth) {
    return candidateDepth > currentDepth;
  }
  return candidate.vStep > current.vStep;
}

ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::FinalizeVStepFromUb(int64_t ubSize, int64_t usedUbBytes,
                                                                                 int64_t coeff) {
  (void)coeff;
  int64_t aDk = Ops::Transformer::CeilAlign(tilingData_.dk, static_cast<uint32_t>(16));  // 16 * 2 = 32B
  BufferProfile selected;
  const std::array<BufferProfile, 3> candidates = {{
      BufferProfile(1, 1, 0, 0, false),
      BufferProfile(1, 2, 0, 0, false),
      BufferProfile(2, 2, 0, 0, false),
  }};
  for (const auto& candidate : candidates) {
    BufferProfile profile;
    if (!EvaluateBufferProfile(ubSize, usedUbBytes, aDk, candidate.stateOutBufferNum, candidate.attnOutBufferNum,
                               profile)) {
      continue;
    }
    if (IsBetterProfile(profile, selected)) {
      selected = profile;
    }
  }
  if (!selected.valid) {
    OP_LOGE(context_->GetNodeName(), "vStep should be bigger than 8, shape is too big");
    return ge::GRAPH_FAILED;
  }

  int64_t queueCoeff = (2 + static_cast<int64_t>(2 * selected.stateOutBufferNum)) * aDk +
                       static_cast<int64_t>(4 * selected.attnOutBufferNum);
  int64_t ubRestBytes = ubSize - ubCalcCtx_.fixedUbBytes - queueCoeff * static_cast<int64_t>(selected.vStep);
  if (ubRestBytes < 0) {
    OP_LOGE(context_->GetNodeName(), "ubRestBytes should be non-negative, but got %ld", ubRestBytes);
    return ge::GRAPH_FAILED;
  }
  tilingData_.ubCalSize = compileInfo_.ubSize;
  tilingData_.vStep = selected.vStep;
  tilingData_.stateOutBufferNum = selected.stateOutBufferNum;
  tilingData_.attnOutBufferNum = selected.attnOutBufferNum;
  tilingData_.ubRestBytes = static_cast<uint32_t>(ubRestBytes);
  return ge::GRAPH_SUCCESS;
}

// CalUbSize now runs an ordered UB rule-chain with explicit intermediate states.
ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::CalUbSize() {
  struct RuleItem {
    const char* name;
    HostRuleFn fn;
  };
  const std::array<RuleItem, 5> ubRules = {{
      {"RuleInitUbCalcContext", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleInitUbCalcContext},
      {"RuleCalcFixedUbBytes", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcFixedUbBytes},
      {"RuleCalcWorkingUbBytes", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcWorkingUbBytes},
      {"RuleCalcVStepCoeff", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleCalcVStepCoeff},
      {"RuleFinalizeVStepFromUb", &FusedGdnL2RecurrentGatedDeltaRuleV310Tiling::RuleFinalizeVStepFromUb},
  }};
  for (const auto& rule : ubRules) {
    OP_CHECK_IF((this->*(rule.fn))() != ge::GRAPH_SUCCESS,
                OP_LOGE(inputParams_.opName, "CalUbSize rule failed: %s", rule.name), return ge::GRAPH_FAILED);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedGdnL2RecurrentGatedDeltaRuleV310TilingFunc(gert::TilingContext* context) {
  OP_CHECK_IF(context == nullptr, OP_LOGE("FusedGdnL2RecurrentGatedDeltaRuleV310", "context is null"),
              return ge::GRAPH_FAILED);
  return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForFusedGdnL2RecurrentGatedDeltaRuleV310(gert::TilingParseContext* context) {
  OP_CHECK_IF(context == nullptr, OP_LOGE("FusedGdnL2RecurrentGatedDeltaRuleV310", "context is null"),
              return ge::GRAPH_FAILED);

  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
              return ge::GRAPH_FAILED);

  auto compileInfoPtr = context->GetCompiledInfo<FusedGdnL2RecurrentGatedDeltaRuleV310CompileInfo>();
  OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
              return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedGdnL2RecurrentGatedDeltaRuleV310)
    .Tiling(FusedGdnL2RecurrentGatedDeltaRuleV310TilingFunc)
    .TilingParse<FusedGdnL2RecurrentGatedDeltaRuleV310CompileInfo>(
        TilingPrepareForFusedGdnL2RecurrentGatedDeltaRuleV310);
}  // namespace optiling
