/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "core/framework/request/dit_request_state.h"
#include "core/framework/sampling/sampling_params.h"
#include "worker.pb.h"

namespace xllm {

void sampling_params_to_proto(
    const RequestSamplingParam& sampling_params,
    proto::RequestSamplingParam* proto_sampling_params);

void sampling_params_from_proto(
    const proto::RequestSamplingParam& proto_sampling_params,
    RequestSamplingParam* sampling_params);

void dit_generation_params_to_proto(
    const DiTGenerationParams& generation_params,
    proto::DiTGenerationParams* proto_generation_params);

void dit_generation_params_from_proto(
    const proto::DiTGenerationParams& proto_generation_params,
    DiTGenerationParams* generation_params);

}  // namespace xllm
