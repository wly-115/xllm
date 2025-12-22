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

#include "minicpmv_input_processor.h"

#include "image/minicpmv_image_processor.h"

namespace xllm {

MiniCPMInputProcessor::MiniCPMInputProcessor(const ModelArgs& args) {
  image_feature_size_ = args.mm_image_feature_size();
  max_slice_nums_ = args.vision_max_slice_nums();
  slice_mode_ = args.mm_slice_mode();
  use_image_id_ = args.mm_use_image_id();
  scale_resolution_ = args.mm_scale_resolution();
  image_processor_ = std::make_unique<MiniCPMVImageProcessor>(args);
}
bool process_multimodal_inputs(const MMInput& mm_inputs,
                               MMData& mm_data) override {
  return image_processor_->process(mm_inputs, mm_data);
}

bool MiniCPMInputProcessor::process(std::string& prompt,
                                    const MMInput& mm_inputs,
                                    MMData& mm_data) {
  bool process_result = image_processor_->process(mm_inputs, mm_data);
  replace_placeholder(prompt, mm_data);
  return process_result;
}

void MiniCPMInputProcessor::replace_placeholder(
    std::string& prompt,
    const MMData& mm_data) override {
  auto image_sizes = mm_data.get_tensor_vec("image_sizes");

  const std::regex pattern(R"(\(<image>[\s\S]*?</image>\))");

  std::sregex_iterator image_tag_begin(prompt.begin(), prompt.end(), pattern);
  std::sregex_iterator image_tag_end;

  if (image_tag_begin == image_tag_end) {
    return;
  }

  std::vector<std::pair<int, int>> image_size_list;
  image_size_list.reserve(image_sizes.size());
  for (auto& image_size : image_sizes) {
    if (image_size.dim() != 1 || image_size.size(0) != 2) {
      const auto& sizes = image_size.sizes();
      LOG(FATAL) << "image_size must be a 1D tensor with 2 "
                    "elements representing height and width;"
                    "now sizes: "
                 << sizes;
    }
    image_size_list.emplace_back(
        std::make_pair(image_size[0].item<int>(), image_size[1].item<int>()));
  }

  std::vector<std::string> text_chunks;
  size_t last_pos = 0;

  for (auto it = image_tag_begin; it != image_tag_end; ++it) {
    auto match = *it;
    text_chunks.push_back(prompt.substr(last_pos, match.position() - last_pos));
    last_pos = match.position() + match.length();
  }

  text_chunks.push_back(prompt.substr(last_pos));

  std::string new_prompt;
  for (size_t i = 0; i < image_size_list.size(); ++i) {
    new_prompt += text_chunks[i];
    new_prompt += get_slice_image_placeholder(image_size_list[i], i);
  }

  new_prompt += text_chunks.back();
  prompt = new_prompt;
}

std::string MiniCPMInputProcessor::get_image_id_placeholder(int idx) const {
  return im_id_start_ + std::to_string(idx) + im_id_end_;
}

std::string MiniCPMInputProcessor::get_grid_placeholder(
    const std::pair<int, int>& grid) const {
  if (grid.first == 0 || grid.second == 0) {
    return "";
  }

  // Prepare the slice placeholder
  std::string slice_placeholder = slice_start_token_;

  // Append the repeated unk_token_
  for (int i = 0; i < image_feature_size_; ++i) {
    slice_placeholder += unk_token_;
  }

  slice_placeholder += slice_end_token_;

  // Use a string to accumulate the result
  std::string grid_placeholder;

  // Loop over the grid and append placeholders
  for (int i = 0; i < grid.second; ++i) {     // Iterate through rows
    for (int j = 0; j < grid.first; ++j) {    // Iterate through columns
      grid_placeholder += slice_placeholder;  // Append the placeholder
    }
    if (i < grid.second - 1) {
      grid_placeholder +=
          "\n";  // Add a newline after each row except the last one
    }
  }

  return grid_placeholder;
}

std::string MiniCPMInputProcessor::get_slice_image_placeholder(
    const std::pair<int, int>& image_size,
    int image_idx = 0,
    int max_slice_nums = -1,
    std::optional<bool> use_image_id_opt = std::nullopt) const {
  if (max_slice_nums < 0) {
    max_slice_nums = max_slice_nums_;
  }

  bool use_image_id =
      use_image_id_opt.has_value() ? use_image_id_opt.value() : use_image_id_;

  assert(max_slice_nums > 0);

  auto grid = MiniCPMVImageProcessor::get_sliced_grid(
      image_size, max_slice_nums, scale_resolution_);

  std::string image_placeholder = im_start_token_;

  for (int i = 0; i < image_feature_size_; ++i) {
    image_placeholder += unk_token_;
  }

  image_placeholder += im_end_token_;

  std::string final_placeholder;

  if (use_image_id) {
    final_placeholder = get_image_id_placeholder(image_idx) + image_placeholder;
  } else {
    final_placeholder = image_placeholder;
  }

  if (slice_mode_) {
    final_placeholder += get_grid_placeholder(grid);
  }

  return final_placeholder;
}

}  // namespace xllm
