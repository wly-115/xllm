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

#include "qwen2_vl_video_processor.h"

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

torch::Tensor Qwen2VLVideoProcessor::sample_frames(
    const VideoMetadata& metadata,
    int temporal_patch_size,
    int min_frames,
    int max_frames,
    int num_frames,
    double set_fps) {
  if (set_fps > 0.0 && num_frames > 0) {
    LOG(FATAL) << "num_frames and fps are mutually exclusive arguments, please "
                  "use only one!";
  }

  double fps = set_fps;

  int total_num_frames = metadata.total_num_frames;

  if (num_frames > 0) {
    double double_num_frames =
        std::round(static_cast<double>(num_frames) / temporal_patch_size) *
        temporal_patch_size;
    num_frames = static_cast<int>(double_num_frames);
  } else if (fps > 0.0) {
    if (metadata.fps <= 0.0) {
      LOG(FATAL)
          << "Asked to sample `fps` frames per second but no video metadata "
             "was provided which is required when sampling with `fps`. ";
    }

    max_frames =
        (std::min(max_frames, total_num_frames) / temporal_patch_size) *
        temporal_patch_size;
    double double_num_frames =
        static_cast<double>(total_num_frames) / metadata.fps * fps;
    double_num_frames = std::min(
        std::min(std::max(double_num_frames, static_cast<double>(min_frames)),
                 static_cast<double>(max_frames)),
        static_cast<double>(total_num_frames));
    double_num_frames = std::floor(double_num_frames / temporal_patch_size) *
                        temporal_patch_size;

    num_frames = static_cast<int>(double_num_frames);
  }

  if (num_frames > total_num_frames) {
    LOG(FATAL) << "Video can't be sampled. The inferred num_frames="
               << num_frames << " exceeds total_num_frames=" << total_num_frames
               << ".";
  }

  if (num_frames > 0) {
    std::vector<int64_t> indices;
    indices.reserve(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      int64_t k = static_cast<int64_t>(
          (static_cast<int64_t>(i) * total_num_frames) / num_frames);
      if (k >= total_num_frames) k = total_num_frames - 1;
      indices.push_back(k);
    }
    return torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong));
  } else {
    return torch::arange(0,
                         static_cast<int64_t>(total_num_frames),
                         torch::TensorOptions().dtype(torch::kLong));
  }
}

torch::Tensor Qwen2VLVideoProcessor::get_video_frames(
    const int64_t origin_frames_num,
    const VideoMetadata& metadata,
    int temporal_patch_size,
    int min_frames,
    int max_frames,
    int num_frames,
    double set_fps) {
  torch::Tensor indices;
  if (do_sample_frame_) {
    indices = this->sample_frames(metadata,
                                  temporal_patch_size_,
                                  min_frames_,
                                  max_frames_,
                                  /*num_frames=*/-1,
                                  /*set_fps=*/2.0);
  } else {
    indices = torch::arange(
        0, origin_frames_num, torch::TensorOptions().dtype(torch::kLong));
  }
  LOG(INFO) << "origin_frames_num " << origin_frames_num;
  LOG(INFO) << "now frames " << indices.size(0);
  return indices;
}

void Qwen2VLVideoProcessor::update_video_metadata(VideoMetadata& metadata,
                                                  torch::Tensor frames_index) {
  auto sampled_total_frames = frames_index.size(0);
  metadata.frame_indices = frames_index;
  metadata.timestamps.clear();
  metadata.timestamps.reserve(static_cast<size_t>(sampled_total_frames));
  double fps_for_ts = (metadata.fps > 0.0) ? metadata.fps : 24.0;
  for (int64_t i = 0; i < sampled_total_frames; ++i) {
    int64_t frame_idx = metadata.frame_indices[i].item<int64_t>();
    metadata.timestamps.push_back(static_cast<double>(frame_idx) / fps_for_ts);
  }

  if (metadata.total_num_frames > 0 && metadata.fps > 0.0) {
    metadata.sampled_fps = double(sampled_total_frames) /
                           double(metadata.total_num_frames) * metadata.fps;
  } else {
    metadata.sampled_fps = fps_for_ts;
  }
}

Qwen2VLVideoProcessor::Qwen2VLVideoProcessor(const ModelArgs& args) {
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

bool Qwen2VLVideoProcessor::process(const MMInput& inputs, MMData& datas) {
  std::vector<torch::Tensor> videos = inputs.get_decode_data(MMType::VIDEO);
  std::vector<VideoMetadata> video_meta_list = inputs.get_video_metadata();

  if (videos.empty() || video_meta_list.empty()) {
    LOG(ERROR) << "no image/video tensor found.";
    return false;
  }

  if (!videos.empty()) {
    if (!this->process_videos(videos, video_meta_list, datas)) {
      LOG(ERROR) << " process video failed.";
      return false;
    }
  }

  return true;
}

bool Qwen2VLVideoProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  std::vector<torch::Tensor> pixel_values;
  std::vector<int64_t> grids;
  int32_t videos_num = videos.size();
  for (int idx = 0; idx < videos_num; ++idx) {
    auto frames_index = get_video_frames(videos[idx].size(0),
                                         video_meta_list[idx],
                                         temporal_patch_size_,
                                         min_frames_,
                                         max_frames_,
                                         /*num_frames=*/-1,
                                         /*set_fps=*/2.0);
    update_video_metadata(video_meta_list[idx], frames_index);
    auto video = videos[idx];
    videos[idx] = video.index_select(/*dim=*/0, frames_index);
  }

  auto [groupd_videos, grouped_indexes] = group_videos_by_shape(videos);
  absl::flat_hash_map<Shape, torch::Tensor> resized_videos_grouped;
  // BTHW
  for (auto& [video_shape, stacked_videos] : groupd_videos) {
    auto resized_height = video_shape[1];
    auto resized_width = video_shape[2];
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
      stacked_videos = ImageUtils::resize_video(
          stacked_videos, {resized_height, resized_width}, resample_, false);
    }
    resized_videos_grouped[video_shape] = stacked_videos;
  }

  auto resized_videos = reorder_videos(resized_videos_grouped, grouped_indexes);

  auto [grouped_resized_videos, grouped_resized_indexes] =
      group_videos_by_shape(resized_videos);

  absl::flat_hash_map<Shape, torch::Tensor> processed_videos_grouped;
  absl::flat_hash_map<Shape, torch::Tensor> processed_grids_grouped;

  for (auto& [shape, stacked_videos] : grouped_resized_videos) {
    auto resized_height = shape[1];
    auto resized_width = shape[2];
    // normalize
    if (do_normalize_) {
      stacked_videos =
          ImageUtils::normalize(stacked_videos, image_mean_, image_std_);
    }

    // rescale
    if (do_rescale_) {
      stacked_videos = ImageUtils::rescale(stacked_videos, rescale_factor_);
    }
    auto num_frames = stacked_videos.size(1);
    auto pad_t = (temporal_patch_size_ - (num_frames % temporal_patch_size_)) %
                 temporal_patch_size_;

    auto patches = stacked_videos;
    if (pad_t > 0) {
      auto last_frame = patches.index({
          torch::indexing::Slice(),  // B: all
          -1                         // T: last
      });
      last_frame = last_frame.repeat({1, pad_t, 1, 1, 1});
      patches = torch::cat({patches, last_frame}, 1);
      LOG(INFO) << "patches size " << patches.sizes();
      LOG(INFO) << "repeats size " << last_frame.sizes();
    }

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

    processed_videos_grouped[shape] = flatten_patches;
    auto grid_thw = torch::tensor({grid_t, grid_h, grid_w}, torch::kLong);
    processed_grids_grouped[shape] =
        torch::tensor({grid_t, grid_h, grid_w}, torch::kLong)
            .unsqueeze(0)
            .expand({batch_size, -1});
  }

  LOG(INFO) << "============= 1";
  const auto& processed_videos =
      reorder_videos(processed_videos_grouped, grouped_resized_indexes);
  LOG(INFO) << "============= 2";
  const auto& processed_grids =
      reorder_videos(processed_grids_grouped, grouped_resized_indexes);

  auto values = torch::cat(processed_videos);
  auto thw = torch::cat(processed_grids).reshape({-1, 3});

  // const size_t videos_num = videos.size();
  // for (size_t i = 0; i < videos_num; ++i) {
  //   auto& vid = videos[i];
  //   auto& metadata = video_meta_list[i];
  //   if (!this->process_video(vid, metadata, pixel_values, grids)) {
  //     return false;
  //   }
  // }

  // auto values = torch::cat(pixel_values);
  // auto thw = torch::tensor(grids).clone().reshape({-1, 3});

  const size_t num_videos = videos.size();
  std::vector<double> second_per_grid;
  second_per_grid.reserve(num_videos);
  for (size_t i = 0; i < num_videos; ++i) {
    const auto& metadata = video_meta_list[i];
    double fps =
        metadata.sampled_fps > 0.0 ? metadata.sampled_fps : metadata.fps;
    double seconds_per_grid = static_cast<double>(temporal_patch_size_) / fps;
    second_per_grid.push_back(seconds_per_grid);
  }
  mm_datas.set_video_metadata(video_meta_list);

  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto second_per_grid_ts = torch::tensor(second_per_grid, opts);

  mm_datas.add(MMType::VIDEO, "video_grid_thw", thw);
  mm_datas.add(MMType::VIDEO, "pixel_values_videos", values);
  mm_datas.add(MMType::VIDEO, "second_per_grid_ts", second_per_grid_ts);

  return true;
}

}  // namespace xllm
