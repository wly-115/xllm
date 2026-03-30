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

#include "processors/models/glm4v_input_processor.h"

namespace xllm {

Glm4VInputProcessor::Glm4VInputProcessor(const ModelArgs& args)
    : image_processor_(args), video_processor_(args) {}

bool Glm4VInputProcessor::process(const MMInput& inputs, MMData& datas) {
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

bool Glm4VInputProcessor::process_images(std::vector<torch::Tensor> images,
                                         MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  for (const auto& img : images) {
    if (!image_processor_.process(img, pixel_values, thw)) {
      LOG(ERROR)
          << "Failed to process image. The shape(channels, height, width) is: "
          << img.sizes();
      return false;
    }

    auto& item = mm_datas.add(MMType::IMAGE);
    item.set_data({{"pixel_values", pixel_values}, {"image_grid_thw", thw}});
  }

  return true;
}

bool Glm4VInputProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  const size_t video_size = videos.size();
  for (size_t i = 0; i < video_size; ++i) {
    auto& vid = videos[i];
    auto& metadata = video_meta_list[i];
    if (!video_processor_.process(vid, metadata, pixel_values, thw)) {
      LOG(ERROR) << "Failed to process video. The shape(num_frames, channels, "
                    "height, width) is: "
                 << vid.sizes();
      return false;
    }

    auto& item = mm_datas.add(MMType::VIDEO);
    item.set_data(
        {{"pixel_values_videos", pixel_values}, {"video_grid_thw", thw}});
    item.set_metadata(metadata);
  }

  return true;
}

}  // namespace xllm
