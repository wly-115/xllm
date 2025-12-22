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

#include "video_processor.cpp"

namespace xllm {

std::pair<absl::flat_hash_map<Shape, torch::Tensor>,
          std::vector<std::pair<Shape, size_t>>>
VideoProcessor::group_videos_by_shape(
    const std::vector<torch::Tensor>& videos) {
  absl::flat_hash_map<Shape, std::vector<torch::Tensor>> temp_groups;
  std::vector<std::pair<Shape, size_t>> grouped_index;
  absl::flat_hash_map<Shape, torch::Tensor> grouped_videos;
  grouped_index.reserve(videos.size());
  // video [TCHW]
  for (auto& video : videos) {
    Shape shape = {video.size(0), video.size(-2), video.size(-1)};

    auto& group = temp_groups[shape];
    size_t pos_in_group = group.size();
    group.push_back(video);
    grouped_index.push_back(std::make_pair(shape, pos_in_group - 1));
  }

  for (auto& [shape, tensors] : temp_groups) {
    grouped_videos[shape] = torch::stack(tensors, 0);
  }
  return std::make_pair(std::move(grouped_videos), std::move(grouped_index));
}

std::vector<torch::Tensor> VideoProcessor::reorder_videos(
    absl::flat_hash_map<Shape, torch::Tensor> grouped_videos,
    std::vector<std::pair<Shape, size_t>> grouped_indexes) {
  std::vector<torch::Tensor> ordered_videos;
  int video_num = grouped_indexes.size();
  ordered_videos.reserve(video_num);

  for (int idx = 0; idx < video_num; idx++) {
    ordered_videos.push_back(grouped_videos[grouped_indexes[idx].first]
                                           [grouped_indexes[idx].second]);
  }
  return ordered_videos;
}

}  // namespace xllm
