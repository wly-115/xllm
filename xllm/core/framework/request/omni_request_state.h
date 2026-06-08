/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <functional>
#include <string>

#include "dit_request_state.h"

namespace xllm {

// Ensemble 输出回调（统一类型，接收 proto NodePayload）
// 前置声明避免循环依赖
namespace proto {
class NodePayload;
}

using OmniOutputFunc = std::function<bool(const proto::NodePayload& output)>;

// Omni 请求状态（P0: 只支持 qwen-image-edit，仅包含 DiT 相关字段）
class OmniRequestState final {
 public:
  OmniRequestState() = default;

  OmniRequestState(DiTInputParams input_params,
                   DiTGenerationParams generation_params,
                   OmniOutputFunc output_func)
      : input_params_(std::move(input_params)),
        generation_params_(std::move(generation_params)),
        output_func_(std::move(output_func)) {}

  // Accessors
  DiTInputParams& input_params() { return input_params_; }
  const DiTInputParams& input_params() const { return input_params_; }

  DiTGenerationParams& generation_params() { return generation_params_; }
  const DiTGenerationParams& generation_params() const {
    return generation_params_;
  }

  OmniOutputFunc& output_func() { return output_func_; }
  const OmniOutputFunc& output_func() const { return output_func_; }

 private:
  DiTInputParams input_params_;
  DiTGenerationParams generation_params_;
  OmniOutputFunc output_func_;
};

}  // namespace xllm
