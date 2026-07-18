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

#include <memory>
#include <vector>

#include "core/framework/ensemble/ensemble_protocol.h"
#include "core/framework/request/dit_request.h"
#include "core/framework/request/dit_request_output.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"

namespace xllm {

template <typename RequestType, typename OutputType>
class EngineDataAdapter {
 public:
  virtual ~EngineDataAdapter() = default;

  virtual std::vector<std::shared_ptr<RequestType>> build_requests(
      const NodeData& input) = 0;

  virtual NodeData convert_output(const NodeData& input,
                                  const std::vector<OutputType>& outputs) = 0;
};

using ArEngineDataAdapter = EngineDataAdapter<Request, RequestOutput>;
using DitEngineDataAdapter = EngineDataAdapter<DiTRequest, DiTRequestOutput>;

}  // namespace xllm
