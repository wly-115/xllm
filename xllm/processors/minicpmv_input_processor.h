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

class MiniCPMInputProcessor : public InputProcessor {
 public:
  MiniCPMInputProcessor(const ModelArgs& args);

  bool process(std::string& prompt, const MMInput& mm_inputs, MMData& mm_data);
  void replace_placeholder(std::string& prompt, const MMData& mm_data);
  bool process_multimodal_inputs(const MMInput& mm_inputs, MMData& mm_data);

 private:
  std::string get_image_id_placeholder(int idx) const;

  std::string get_grid_placeholder(const std::pair<int, int>& grid) const;

  std::string get_slice_image_placeholder(
      const std::pair<int, int>& image_size,
      int image_idx = 0,
      int max_slice_nums = -1,
      std::optional<bool> use_image_id_opt = std::nullopt) const;

 private:
  const std::string im_start_token_ = "<image>";
  const std::string im_end_token_ = "</image>";
  const std::string slice_start_token_ = "<slice>";
  const std::string slice_end_token_ = "</slice>";
  const std::string unk_token_ = "<unk>";
  const std::string im_id_start_ = "<image_id>";
  const std::string im_id_end_ = "</image_id>";

  bool slice_mode_;
  bool use_image_id_;
  int max_slice_nums_;
  int image_feature_size_;
  int scale_resolution_;
};
}  // namespace xllm