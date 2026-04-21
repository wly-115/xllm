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

#include "core/framework/request/mm_data_visitor.h"
#include "processors/common/multimodal_utils.h"

namespace xllm {

Glm4VInputProcessor::Glm4VInputProcessor(const ModelArgs& args)
    : image_processor_(args), video_processor_(args) {}

bool Glm4VInputProcessor::process(const MMInput& inputs, MMData& datas) {
  std::vector<torch::Tensor> images = inputs.get_decode_data(MMType::IMAGE);
  std::vector<torch::Tensor> videos = inputs.get_decode_data(MMType::VIDEO);
  std::vector<VideoMetadata> video_meta_list = inputs.get_video_metadata();

  for (const auto& input_item : inputs) {
    if (input_item.has_type(MMType::IMAGE) && !input_item.is_embedding()) {
      datas.add(MMType::IMAGE);
    }
    if (input_item.has_type(MMType::VIDEO)) {
      datas.add(MMType::VIDEO);
    }
  }

  PreprocessedMMItems items;
  if (!images.empty() && !this->process_images(images, items.image_items)) {
    LOG(ERROR) << " process image failed.";
    return false;
  }
  if (!videos.empty() &&
      !this->process_videos(videos, video_meta_list, items.video_items)) {
    LOG(ERROR) << " process video failed.";
    return false;
  }

  PreprocessOutputScatterVisitor scatter(items);
  CHECK(datas.foreach (scatter))
      << "scatter preprocess outputs failed during visit.";
  CHECK(scatter.finish()) << "scatter preprocess outputs count mismatch.";

  return true;
}

bool Glm4VInputProcessor::process_images(
    const std::vector<torch::Tensor>& images,
    std::vector<MMDataItem>& image_items) {
  std::vector<torch::Tensor> pixel_values_list(images.size());
  std::vector<torch::Tensor> thw_list(images.size());

  const auto image_buckets = group_images_by_shape(images);
  for (const auto& image_bucket : image_buckets) {
    const ImageBatchBucket& bucket = image_bucket.second;
    torch::Tensor batch_images = torch::stack(bucket.images);
    torch::Tensor batch_pixel_values;
    torch::Tensor batch_thw;
    if (!image_processor_.process_batch(
            batch_images, batch_pixel_values, batch_thw)) {
      LOG(ERROR)
          << "Failed to process image batch. The shape(channels, height, "
             "width) is: "
          << batch_images[0].sizes();
      return false;
    }

    std::vector<torch::Tensor> bucket_pixel_values =
        batch_pixel_values.unbind(0);
    std::vector<torch::Tensor> bucket_thw = batch_thw.unbind(0);
    const size_t bucket_size = bucket.indices.size();
    for (size_t index = 0; index < bucket_size; ++index) {
      const size_t output_index = bucket.indices[index];
      pixel_values_list[output_index] = std::move(bucket_pixel_values[index]);
      thw_list[output_index] = std::move(bucket_thw[index]);
    }
  }

  image_items.clear();
  image_items.reserve(images.size());
  for (size_t index = 0; index < images.size(); ++index) {
    image_items.emplace_back(MMType::IMAGE,
                             MMDict{{"pixel_values", pixel_values_list[index]},
                                    {"image_grid_thw", thw_list[index]}});
  }

  return true;
}

bool Glm4VInputProcessor::process_videos(
    const std::vector<torch::Tensor>& videos,
    const std::vector<VideoMetadata>& video_meta_list,
    std::vector<MMDataItem>& video_items) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  video_items.clear();
  video_items.reserve(videos.size());
  const size_t video_size = videos.size();
  for (size_t i = 0; i < video_size; ++i) {
    const auto& vid = videos[i];
    auto metadata = video_meta_list[i];
    if (!video_processor_.process(vid, metadata, pixel_values, thw)) {
      LOG(ERROR) << "Failed to process video. The shape(num_frames, channels, "
                    "height, width) is: "
                 << vid.sizes();
      return false;
    }

    video_items.emplace_back(
        MMType::VIDEO,
        MMDict{{"pixel_values_videos", pixel_values}, {"video_grid_thw", thw}},
        metadata);
  }

  return true;
}

}  // namespace xllm
