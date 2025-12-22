
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

class GLM4VInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  GLM4VInputProcessor(const ModelArgs& args);
  bool process(std::string& prompt, const MMInput& mm_inputs, MMData& mm_data);
  void replace_placeholder(std::string& prompt, const MMData& mm_data);
  bool process_multimodal_inputs(const MMInput& mm_inputs, MMData& mm_data);

 private:
  std::pair<TokenType, size_t> find_vision_token(const std::string& prompt,
                                                 size_t begin);

  std::vector<double> build_timestamps(const std::vector<double>& timestamps,
                                       size_t num_frames);

  std::string format_timestamp_str(double timestamp);

 private:
  const std::string image_token_ = "<|image|>";
  const std::string video_token_ = "<|video|>";

  const std::string begin_of_image_token_ = "<|begin_of_image|>";
  const std::string end_of_image_token_ = "<|end_of_image|>";

  int merge_size_ = 0;
};
}  // namespace xllm