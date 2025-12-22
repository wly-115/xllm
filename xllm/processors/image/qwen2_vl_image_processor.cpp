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

#include "qwen2_vl_image_processor.h"

#include "vision_util.h"

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
  size_ = {{"longest_edge", 12845056}, {"shortest_edge", 3136}};

  // fuse image mean/std and rescale_factor
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

bool Qwen2VLImageProcessor::process(const MMInput& inputs, MMData& datas) {
  std::vector<torch::Tensor> images = inputs.get_decode_data(MMType::IMAGE);
  std::vector<torch::Tensor> videos = inputs.get_decode_data(MMType::VIDEO);
  std::vector<VideoMetadata> video_meta_list = inputs.get_video_metadata();

  if (images.empty() && (videos.empty() || video_meta_list.empty())) {
    LOG(ERROR) << "no image/video tensor found.";
    return false;
  }

  if (!images.empty()) {
    if (!this->process_images(images, datas)) {
      LOG(ERROR) << " process image failed.";
      return false;
    }
  }

  if (!videos.empty()) {
    if (!this->process_videos(videos, video_meta_list, datas)) {
      LOG(ERROR) << " process video failed.";
      return false;
    }
  }

  return true;
}

bool Qwen2VLImageProcessor::process_images(std::vector<torch::Tensor> images,
                                           MMData& mm_datas) {
  std::vector<torch::Tensor> pixel_values;
  std::vector<int64_t> grids;

  auto [groupd_images, grouped_indexes] = group_images_by_shape(images);
  absl::flat_hash_map<Shape, torch::Tensor> resized_images_grouped;

  for (auto& [shape, stacked_images] : groupd_images) {
    auto resized_height = shape[0];
    auto resized_width = shape[1];
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
      stacked_images = ImageUtils::resize(
          stacked_images, {resized_height, resized_width}, resample_, false);
    }
    resized_images_grouped[shape] = stacked_images;
  }

  auto resized_images = reorder_images(resized_images_grouped, grouped_indexes);

  auto [grouped_resized_images, grouped_resized_indexes] =
      group_images_by_shape(resized_images);

  absl::flat_hash_map<Shape, torch::Tensor> processed_images_grouped;
  absl::flat_hash_map<Shape, torch::Tensor> processed_grids_grouped;
  for (auto& [shape, stacked_images] : grouped_resized_images) {
    auto resized_height = shape[0];
    auto resized_width = shape[1];
    // normalize
    if (do_normalize_) {
      stacked_images =
          ImageUtils::normalize(stacked_images, image_mean_, image_std_);
    }

    // rescale
    if (do_rescale_) {
      stacked_images = ImageUtils::rescale(stacked_images, rescale_factor_);
    }
    auto patches = stacked_images.unsqueeze(1);
    auto repeats = patches.slice(1, -1, patches.size(1))
                       .repeat({1, temporal_patch_size_ - 1, 1, 1, 1});
    LOG(INFO) << "patches size " << patches.sizes();
    LOG(INFO) << "repeats size " << repeats.sizes();
    patches = torch::cat({patches, repeats}, 1);

    auto patches_shape = patches.sizes();
    auto batch_size = patches_shape[0];
    auto channel = patches_shape[2];
    auto grid_t = patches_shape[1] / temporal_patch_size_;

    auto grid_h = resized_height / patch_size_;
    auto grid_w = resized_width / patch_size_;
    // #(B, grid_t, gh, gw, mh, mw, C, tp, ph, pw)
    LOG(INFO) << "grid_t " << grid_t;

    patches = patches.view({batch_size,
                            grid_t,
                            temporal_patch_size_,
                            channel,
                            grid_h / merge_size_,
                            merge_size_,
                            patch_size_,
                            grid_w / merge_size_,
                            merge_size_,
                            patch_size_});
    LOG(INFO) << "patches size " << patches.sizes();
    patches = patches.permute({0, 1, 4, 7, 5, 8, 3, 2, 6, 9});

    auto flatten_patches = patches.reshape(
        {batch_size,
         grid_t * grid_h * grid_w,
         channel * temporal_patch_size_ * patch_size_ * patch_size_});

    processed_images_grouped[shape] = flatten_patches;
    auto grid_thw = torch::tensor({grid_t, grid_h, grid_w}, torch::kLong);
    processed_grids_grouped[shape] =
        torch::tensor({grid_t, grid_h, grid_w}, torch::kLong)
            .unsqueeze(0)
            .expand({batch_size, -1});
  }
  LOG(INFO) << "============= 1";
  const auto& processed_images =
      reorder_images(processed_images_grouped, grouped_resized_indexes);
  LOG(INFO) << "============= 2";
  const auto& processed_grids =
      reorder_images(processed_grids_grouped, grouped_resized_indexes);

  auto values = torch::cat(processed_images);
  auto thw = torch::cat(processed_grids).reshape({-1, 3});
  LOG(INFO) << "values " << values.sizes();
  LOG(INFO) << "thw " << thw.sizes();

  mm_datas.add(MMType::IMAGE, "image_grid_thw", thw);
  mm_datas.add(MMType::IMAGE, "pixel_values", values);
  return true;
}

}  // namespace xllm
