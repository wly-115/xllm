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
#include "input_processor.h"

namespace xllm {
struct ModelArgs;

class Qwen2_5_VLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  Qwen2_5_VLInputProcessor(const ModelArgs& args);
  bool process(std::string& prompt, const MMInput& mm_inputs, MMData& mm_data);
  void replace_placeholder(std::string& prompt, const MMData& mm_data);
  bool process_multimodal_inputs(const MMInput& mm_inputs, MMData& mm_data);

 private:
  std::pair<TokenType, size_t> find_vision_token(const std::string& prompt,
                                                 size_t begin);

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  int merge_size_ = 0;
};
}  // namespace xllm