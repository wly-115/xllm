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

#include "clip_input_processor.h"

#include "image/clip_image_processor.h"

namespace xllm {

class CLIPVLInputProcessor : public InputProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  explicit CLIPVLInputProcessor(const ModelArgs& args) {
    merge_size_ = args.mm_image_merge_size();
    image_processor_ = std::make_unique<>
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

  void CLIPVLInputProcessor::replace_placeholder(
      std::string& prompt,
      const MMData& mm_data) override {
    torch::Tensor image_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();
    torch::Tensor video_grid_thw;
    if (auto res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();
    if (!image_grid_thw.defined() && !video_grid_thw.defined()) return;
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
        total_video_token +=
            video_grid_thw[idx].prod().item<int>() / merge_length;
    }
    size_t total_token_len = total_image_token * image_token_.size() +
                             total_video_token * video_token_.size();
    std::string data;
    data.reserve(prompt.size() + total_token_len);
    int image_index = 0;
    int video_index = 0;
    const torch::Tensor* grid_thw = nullptr;
    const std::string* token = nullptr;
    int* index = 0;
    size_t begin = 0;
    auto pair = _find_vision_token(prompt, begin);
    while (pair.second != std::string::npos) {
      data.append(prompt, begin, pair.second - begin);
      if (pair.first == TokenType::IMAGE) {
        grid_thw = &image_grid_thw;
        token = &image_token_;
        index = &image_index;
      } else if (pair.first == TokenType::VIDEO) {
        grid_thw = &video_grid_thw;
        token = &video_token_;
        index = &video_index;
      } else {
        assert(false);
      }
      auto token_num = (*grid_thw)[(*index)].prod().item<int>() / merge_length;
      while (token_num--) data.append(*token);
      ++(*index);
      begin = pair.second + token->size();
      pair = _find_vision_token(prompt, begin);
    }
    if (begin < prompt.size()) data.append(prompt, begin, std::string::npos);
    prompt = std::move(data);
  }

 private:
  std::pair<TokenType, size_t> _find_vision_token(const std::string& prompt,
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

 private:
  const std::string image_token_ = "<|image_pad|>";
  const std::string video_token_ = "<|video_pad|>";
  int merge_size_ = 0;
};
}  // namespace xllm