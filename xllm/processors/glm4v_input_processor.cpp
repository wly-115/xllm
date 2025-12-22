
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
#include "glm4v_input_processor.h"

#include "image/glm4v_image_processor.h"
#include "video/glm4v_video_processor.h"

namespace xllm {
GLM4VInputProcessor::GLM4VInputProcessor(const ModelArgs& args) {
  merge_size_ = args.mm_image_merge_size();
  image_processor_ = std::make_unique<Glm4VImageProcessor>(args);
  video_processor_ = std::make_unique<Glm4VVideoProcessor>(args);
}

bool process_multimodal_inputs(const MMInput& mm_inputs,
                               MMData& mm_data) override {
  return image_processor_->process(mm_inputs, mm_data) &&
         video_processor_->process(mm_inputs, mm_data);
}

bool process(std::string& prompt,
             const MMInput& mm_inputs,
             MMData& mm_data) override {
  bool process_result = image_processor_->process(mm_inputs, mm_data) &&
                        video_processor_->process(mm_inputs, mm_data);
  replace_placeholder(prompt, mm_data);
  return process_result;
}
void GLM4VInputProcessor::replace_placeholder(std::string& prompt,
                                              const MMData& mm_data) override {
  torch::Tensor image_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
    image_grid_thw = res.value();

  torch::Tensor video_grid_thw;
  if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
    video_grid_thw = res.value();

  if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;

  const auto& video_metadata = mm_data.get_video_metadata();
  if (video_metadata.size() > 0) {
    CHECK(video_metadata.size() ==
          static_cast<size_t>(video_grid_thw.sizes()[0]));
  }

  auto merge_length = merge_size_ * merge_size_;
  int total_image_token = 0;

  if (image_grid_thw.defined()) {
    auto count = image_grid_thw.sizes()[0];
    for (int idx = 0; idx < count; ++idx)
      total_image_token +=
          image_grid_thw[idx].prod().item<int>() / merge_length;
  }

  int total_video_token = 0;
  if (video_grid_thw.defined()) {
    auto count = video_grid_thw.sizes()[0];
    for (int idx = 0; idx < count; ++idx)
      total_video_token += video_grid_thw[idx].prod().item<int>() /
                           merge_length / video_grid_thw[idx][0].item<int>();
  }

  size_t total_token_len = total_image_token * image_token_.size() +
                           total_video_token * image_token_.size();
  std::string data;
  data.reserve(prompt.size() + total_token_len);

  int image_index = 0;
  int video_index = 0;

  size_t begin = 0;
  auto pair = find_vision_token(prompt, begin);

  while (pair.second != std::string::npos) {
    data.append(prompt, begin, pair.second - begin);

    if (pair.first == TokenType::IMAGE) {
      auto token_num =
          image_grid_thw[image_index].prod().item<int>() / merge_length;
      while (token_num--) data.append(image_token_);

      image_index++;
      begin = pair.second + image_token_.size();
    } else if (pair.first == TokenType::VIDEO) {
      auto num_frames = video_grid_thw[video_index][0].item<int>();
      auto timestamps = video_metadata[video_index].timestamps;
      CHECK(!timestamps.empty());

      auto selected = build_timestamps(timestamps, num_frames);
      auto token_num = video_grid_thw[video_index].prod().item<int>() /
                       merge_length / num_frames;

      for (size_t idx = 0; idx < num_frames; ++idx) {
        data.append(begin_of_image_token_);

        auto num = token_num;
        while (num--) data.append(image_token_);

        data.append(end_of_image_token_);
        data.append(format_timestamp_str(selected[idx]));
      }

      video_index++;
      begin = pair.second + video_token_.size();
    } else {
      assert(false);
    }

    pair = find_vision_token(prompt, begin);
  }

  if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);

  prompt = std::move(data);
}
bool process(std::string& prompt,
             const MMInput& mm_inputs,
             MMData& mm_data) override {
  bool process_result = image_processor_->process(mm_inputs, mm_data) &&
                        video_processor_->process(mm_inputs, mm_data);
  replace_placeholder(prompt, mm_data);
  return process_result;
}

std::pair<TokenType, size_t> GLM4VInputProcessor::find_vision_token(
    const std::string& prompt,
    size_t begin) {
  auto img_pos = prompt.find(image_token_, begin);
  auto vid_pos = prompt.find(video_token_, begin);

  if (img_pos == std::string::npos && vid_pos == std::string::npos)
    return {TokenType::INVALID, std::string::npos};
  else if (vid_pos == std::string::npos)
    return {TokenType::IMAGE, img_pos};
  else if (img_pos == std::string::npos)
    return {TokenType::VIDEO, vid_pos};
  else
    return img_pos < vid_pos ? std::make_pair(TokenType::IMAGE, img_pos)
                             : std::make_pair(TokenType::VIDEO, vid_pos);
}

std::vector<double> GLM4VInputProcessor::build_timestamps(
    const std::vector<double>& timestamps,
    size_t num_frames) {
  std::vector<double> vec;
  vec.reserve(num_frames);

  for (size_t i = 0; i < timestamps.size(); i += 2) {
    vec.push_back(timestamps[i]);
    if (vec.size() == num_frames) break;
  }

  while (vec.size() < num_frames) {
    vec.push_back(vec.back());
  }

  return vec;
}

std::string GLM4VInputProcessor::format_timestamp_str(double timestamp) {
  char buffer[32];
  sprintf(buffer, "%.1f seconds", timestamp);
  return buffer;
}

}  // namespace xllm