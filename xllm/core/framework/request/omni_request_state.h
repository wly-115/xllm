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

#include <torch/torch.h>

#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "core/framework/request/dit_request_state.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {

class NodeResult;

using OmniOutputFunc = std::function<bool(const NodeResult& result)>;

struct OmniRequestInput {
  std::string prompt;
  std::string negative_prompt;
  torch::Tensor image;
};

struct OmniRequestParams {
  std::optional<RequestSamplingParam> sampling_params;
  std::optional<DiTGenerationParams> dit_generation_params;
};

class OmniRequestState final {
 public:
  OmniRequestState() = default;

  OmniRequestState(OmniRequestInput input,
                   OmniRequestParams params,
                   OmniOutputFunc output_func)
      : input_(std::move(input)),
        params_(std::move(params)),
        output_func_(std::move(output_func)) {}

  OmniRequestInput& input() { return input_; }
  const OmniRequestInput& input() const { return input_; }

  OmniRequestParams& params() { return params_; }
  const OmniRequestParams& params() const { return params_; }

  OmniOutputFunc& output_func() { return output_func_; }
  const OmniOutputFunc& output_func() const { return output_func_; }

 private:
  OmniRequestInput input_;
  OmniRequestParams params_;
  OmniOutputFunc output_func_;
};

}  // namespace xllm
