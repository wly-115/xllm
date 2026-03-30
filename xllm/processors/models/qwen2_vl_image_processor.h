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

#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/common/base_image_processor.h"

namespace xllm {

class Qwen2VLImageProcessor : protected BaseImageProcessor {
 public:
  using BaseImageProcessor::resize;

  explicit Qwen2VLImageProcessor(const ModelArgs& args);

  bool process(torch::Tensor image,
               torch::Tensor& pixel_values,
               torch::Tensor& thw) const;

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  std::vector<double> image_mean_;
  std::vector<double> image_std_;

  int max_pixels_ = 12845056;
  int min_pixels_ = 3136;

  int merge_size_ = 2;
  int patch_size_ = 14;

  int resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  int temporal_patch_size_ = 2;
};

}  // namespace xllm
