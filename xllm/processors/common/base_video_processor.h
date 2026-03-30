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

#include "processors/common/base_image_processor.h"

namespace xllm {

class BaseVideoProcessor : protected BaseImageProcessor {
 public:
  virtual ~BaseVideoProcessor() = default;

 protected:
  void validateVideoTensor(const torch::Tensor& video) const;
  torch::Tensor applyFrameTransforms(const torch::Tensor& video,
                                     const std::vector<int64_t>& size,
                                     bool do_resize,
                                     int resample,
                                     bool antialias,
                                     bool do_normalize,
                                     const std::vector<double>& mean,
                                     const std::vector<double>& std,
                                     bool do_rescale,
                                     double scale) const;
  torch::Tensor temporalPad(const torch::Tensor& video,
                            int temporal_patch_size) const;
};

}  // namespace xllm
