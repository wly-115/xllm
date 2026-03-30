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

#include <torch/torch.h>

#include <vector>

#include "core/util/tensor_helper.h"
#include "processors/core/multimodal_input_processor.h"
#include "processors/models/minicpmv_image_processor.h"

namespace xllm {

class MiniCPMVInputProcessor : public MultimodalInputProcessor {
 public:
  MiniCPMVInputProcessor(const ModelArgs& args);
  ~MiniCPMVInputProcessor() override = default;

  static std::pair<int, int> get_sliced_grid(
      const std::pair<int, int>& original_size,
      int max_slice_nums,
      int scale_resolution,
      bool never_split = false);

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);

 private:
  MiniCPMVImageProcessor image_processor_;
};

}  // namespace xllm
