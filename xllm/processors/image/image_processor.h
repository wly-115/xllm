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

#include <absl/container/flat_hash_map.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/request/mm_input.h"

namespace xllm {
using Shape = std::vector<int64_t>;

class ImageProcessor {
 public:
  virtual ~ImageProcessor() = default;

  virtual bool process(const MMInput& mm_inputs, MMData& mm_datas) = 0;
  virtual torch::Tensor resize(const torch::Tensor& image,
                               const std::vector<int64_t>& size,
                               int resample,
                               bool antialias = true);
  virtual torch::Tensor centerCrop(const torch::Tensor& image,
                                   const std::pair<int, int>& cropSize);
  virtual torch::Tensor rescale(const torch::Tensor& image, double scale);
  virtual torch::Tensor normalize(const torch::Tensor& image,
                                  const std::vector<double>& mean,
                                  const std::vector<double>& std);

  virtual std::pair<absl::flat_hash_map<Shape, torch::Tensor>,
                    std::vector<std::pair<Shape, size_t>>>
  group_images_by_shape(const std::vector<torch::Tensor>& images);

  virtual std::vector<torch::Tensor> reorder_images(
      absl::flat_hash_map<Shape, torch::Tensor> grouped_images,
      std::vector<std::pair<Shape, size_t>> grouped_indexes);

  virtual std::pair<absl::flat_hash_map<Shape, torch::Tensor>,
                    std::vector<std::pair<Shape, size_t>>>
  group_videos_by_shape(const std::vector<torch::Tensor>& videos);

  virtual std::vector<torch::Tensor> reorder_videos(
      absl::flat_hash_map<Shape, torch::Tensor> grouped_videos,
      std::vector<std::pair<Shape, size_t>> grouped_indexes);
};
}  // namespace xllm
