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

#include "processors/models/minicpmv_input_processor.h"

namespace xllm {

MiniCPMVInputProcessor::MiniCPMVInputProcessor(const ModelArgs& args)
    : image_processor_(args) {}

bool MiniCPMVInputProcessor::process(const MMInput& mm_inputs,
                                     MMData& mm_datas) {
  std::vector<torch::Tensor> images = mm_inputs.get_decode_data(MMType::IMAGE);
  if (images.empty()) {
    LOG(ERROR) << " image tensor not found.";
    return false;
  }

  if (!this->process_images(images, mm_datas)) {
    LOG(ERROR) << " process image failed.";
    return false;
  }

  return true;
}

bool MiniCPMVInputProcessor::process_images(std::vector<torch::Tensor> images,
                                            MMData& mm_datas) {
  std::vector<torch::Tensor> new_images;
  std::vector<torch::Tensor> tgt_sizes;

  const size_t image_size = images.size();
  new_images.reserve(image_size);
  tgt_sizes.reserve(image_size);

  for (const auto& image : images) {
    new_images.clear();
    tgt_sizes.clear();

    if (!image_processor_.process(image, new_images, tgt_sizes)) {
      LOG(ERROR)
          << "Failed to process image. The shape(channels, height, width) is: "
          << image.sizes();
      return false;
    }

    // image shape: [C, H, W]
    const auto& image_size = image.sizes();
    int orig_w = image_size[2];
    int orig_h = image_size[1];

    auto image_sizes = torch::tensor({orig_w, orig_h}, torch::kInt64);
    auto tgt_tensor = torch::stack(tgt_sizes);

    auto& item = mm_datas.add(MMType::IMAGE);
    item.set_data({{"pixel_values", new_images},
                   {"image_sizes", image_sizes},
                   {"tgt_sizes", tgt_tensor}});
  }

  return true;
}

std::pair<int, int> MiniCPMVInputProcessor::get_sliced_grid(
    const std::pair<int, int>& original_size,
    int max_slice_nums,
    int scale_resolution,
    bool never_split) {
  return MiniCPMVImageProcessor::get_sliced_grid(
      original_size, max_slice_nums, scale_resolution, never_split);
}

}  // namespace xllm
