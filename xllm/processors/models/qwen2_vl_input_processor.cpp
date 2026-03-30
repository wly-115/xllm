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

#include "processors/models/qwen2_vl_input_processor.h"

namespace xllm {

Qwen2VLInputProcessor::Qwen2VLInputProcessor(const ModelArgs& args)
    : image_processor_(args), video_processor_(args) {}

bool Qwen2VLInputProcessor::process(const MMInput& inputs, MMData& datas) {
  for (const auto& input_item : inputs) {
    std::vector<torch::Tensor> images;
    std::vector<EmbeddingInput> images_embedding;
    std::vector<torch::Tensor> videos;
    std::vector<VideoMetadata> video_meta_list;
    std::vector<torch::Tensor> audios;
    std::vector<AudioMetadata> audio_meta_list;

    if (input_item.has_type(MMType::IMAGE)) {
      if (input_item.is_embedding()) {
        images_embedding.push_back(input_item.embedding);
      } else {
        images.push_back(input_item.decode_image);
      }
    }
    if (input_item.has_type(MMType::VIDEO)) {
      videos.push_back(input_item.decode_video);
      video_meta_list.push_back(input_item.video_meta);
    }
    if (input_item.has_type(MMType::AUDIO)) {
      audios.push_back(input_item.decode_audio);
      audio_meta_list.push_back(input_item.audio_meta);
    }

    if (images_embedding.empty() && images.empty() &&
        (videos.empty() || video_meta_list.empty()) &&
        (audios.empty() || audio_meta_list.empty())) {
      LOG(ERROR) << "no image/video/audio tensor or embedding found.";
      return false;
    }

    if (!images_embedding.empty()) {
      if (!this->process_images_embedding(images_embedding, datas)) {
        LOG(ERROR) << " process embedding failed.";
        return false;
      }
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
  }
  return true;
}

bool Qwen2VLInputProcessor::process_images_embedding(
    const std::vector<EmbeddingOutput>& images_embedding,
    MMData& mm_datas) {
  for (auto& output : images_embedding) {
    auto& item = mm_datas.add(MMType::IMAGE);
    MMDict data;
    const auto& embeds = output.embedding.chunk(/*chunks*/ 4, /*dim*/ 0);
    data["embedding"] = embeds[0];
    for (size_t i = 0; i < 3; ++i) {
      data[std::string("embedding|deepstack_") + std::to_string(i)] =
          embeds[i + 1];
    }
    for (const auto& [key, value] : output.metadata) {
      data[key] = value;
    }
    item.set_data(data);
  }
  return true;
}

bool Qwen2VLInputProcessor::process_images(std::vector<torch::Tensor> images,
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

bool Qwen2VLInputProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32);

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

    double fps =
        metadata.sampled_fps > 0.0 ? metadata.sampled_fps : metadata.fps;
    double seconds_per_grid =
        static_cast<double>(video_processor_.temporal_patch_size()) / fps;
    auto second_per_grid_ts = torch::tensor({seconds_per_grid}, opts);

    auto& item = mm_datas.add(MMType::VIDEO);
    item.set_data({{"pixel_values_videos", pixel_values},
                   {"video_grid_thw", thw},
                   {"second_per_grid_ts", second_per_grid_ts}});

    item.set_metadata(metadata);
  }

  return true;
}

}  // namespace xllm
