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

#include "processors/models/qwen2_vl_image_processor.h"

#include "processors/common/transforms.h"

namespace xllm {

namespace {

using Size = std::pair<int, int>;

std::optional<Size> smart_resize(int height,
                                 int width,
                                 int factor = 28,
                                 int min_pixels = 56 * 56,
                                 int max_pixels = 14 * 14 * 4 * 1280) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200";
    return std::nullopt;
  }

  int h_bar =
      static_cast<int>(std::round(height / static_cast<double>(factor))) *
      factor;
  int w_bar =
      static_cast<int>(std::round(width / static_cast<double>(factor))) *
      factor;

  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
    h_bar = static_cast<int>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
    h_bar = static_cast<int>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}

}  // namespace

Qwen2VLImageProcessor::Qwen2VLImageProcessor(const ModelArgs& args) {
  image_mean_ = args.mm_image_normalize_mean();
  image_std_ = args.mm_image_normalize_std();
  if (args.mm_image_max_pixels() && args.mm_image_min_pixels()) {
    min_pixels_ = args.mm_image_min_pixels();
    max_pixels_ = args.mm_image_max_pixels();
  } else if (args.mm_image_shortest_edge() && args.mm_image_longest_edge()) {
    min_pixels_ = args.mm_image_shortest_edge();
    max_pixels_ = args.mm_image_longest_edge();
  }
  patch_size_ = args.mm_image_patch_size();
  temporal_patch_size_ = args.mm_image_temporal_patch_size();
  merge_size_ = args.mm_image_merge_size();

  if (do_rescale_ && do_normalize_) {
    for (auto& item : image_mean_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : image_std_) {
      item = item * (1.0 / rescale_factor_);
    }

    do_rescale_ = false;
  }
}

bool Qwen2VLImageProcessor::process(torch::Tensor image,
                                    torch::Tensor& pixel_values,
                                    torch::Tensor& thw) const {
  auto shape = image.sizes();
  auto resized_height = shape[1];
  auto resized_width = shape[2];

  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             min_pixels_,
                             max_pixels_);
    if (!size) {
      return false;
    }

    std::tie(resized_height, resized_width) = *size;
    image = transforms::resize(
        image, {resized_height, resized_width}, resample_, false);
  }

  if (do_normalize_) {
    image = transforms::normalize(image, image_mean_, image_std_);
  }

  if (do_rescale_) {
    image = transforms::rescale(image, rescale_factor_);
  }

  auto patches = torch::stack({image}, 0);
  auto repeats =
      patches[-1].unsqueeze(0).repeat({temporal_patch_size_ - 1, 1, 1, 1});
  patches = torch::cat({patches, repeats}, 0);
  shape = patches.sizes();

  auto channel = shape[1];
  auto grid_t = shape[0] / temporal_patch_size_;
  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;

  patches = patches.view({grid_t,
                          temporal_patch_size_,
                          channel,
                          grid_h / merge_size_,
                          merge_size_,
                          patch_size_,
                          grid_w / merge_size_,
                          merge_size_,
                          patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).clone().reshape({-1, 3});

  return true;
}

}  // namespace xllm
