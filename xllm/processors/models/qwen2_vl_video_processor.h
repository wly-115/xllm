/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <unordered_map>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/request/mm_input.h"
#include "processors/common/base_video_processor.h"

namespace xllm {

class Qwen2VLVideoProcessor : protected BaseVideoProcessor {
 public:
  explicit Qwen2VLVideoProcessor(const ModelArgs& args);

  bool process(torch::Tensor origin_video,
               VideoMetadata& metadata,
               torch::Tensor& pixel_values,
               torch::Tensor& thw) const;
  int temporal_patch_size() const { return temporal_patch_size_; }

 private:
  torch::Tensor sample_frames(const VideoMetadata& metadata,
                              int temporal_patch_size,
                              int min_frames,
                              int max_frames,
                              int num_frames = -1,
                              double set_fps = -1.0) const;

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  std::vector<double> image_mean_;
  std::vector<double> image_std_;

  int merge_size_ = 2;
  int patch_size_ = 14;

  int resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  std::unordered_map<std::string, int> size_;
  int temporal_patch_size_ = 2;

  bool do_sample_frame_ = true;
  int min_frames_ = 4;
  int max_frames_ = 768;
};

}  // namespace xllm
