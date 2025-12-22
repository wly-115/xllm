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

namespace xllm {

class ImageUtils {
 public:
  ImageUtils() = delete;

  static torch::Tensor resize(const torch::Tensor& image,
                              const std::vector<int64_t>& size,
                              int resample,
                              bool antialias = true);
  static torch::Tensor resize_video(const torch::Tensor& video,  //[B,]
                                    const std::vector<int64_t>& size,
                                    int resample,
                                    bool antialias = true);

  static torch::Tensor centerCrop(const torch::Tensor& image,
                                  const std::pair<int, int>& crop_size);

  static torch::Tensor rescale(const torch::Tensor& image, double scale);

  static torch::Tensor normalize(const torch::Tensor& image,
                                 const std::vector<double>& mean,
                                 const std::vector<double>& std);
};

}  // namespace xllm