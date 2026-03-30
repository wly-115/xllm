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

#include "processors/common/base_video_processor.h"

namespace xllm {

void BaseVideoProcessor::validateVideoTensor(const torch::Tensor& video) const {
  if (video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }
}

torch::Tensor BaseVideoProcessor::applyFrameTransforms(
    const torch::Tensor& video,
    const std::vector<int64_t>& size,
    bool do_resize,
    int resample,
    bool antialias,
    bool do_normalize,
    const std::vector<double>& mean,
    const std::vector<double>& std,
    bool do_rescale,
    double scale) const {
  validateVideoTensor(video);

  std::vector<torch::Tensor> out_frames;
  out_frames.reserve(video.size(0));
  auto frames = video.unbind(0);

  for (auto& frame : frames) {
    if (do_resize) {
      frame = resize(frame, size, resample, antialias);
    }
    if (do_normalize) {
      frame = normalize(frame, mean, std);
    }
    if (do_rescale) {
      frame = rescale(frame, scale);
    }
    out_frames.push_back(frame);
  }

  return torch::stack(out_frames);
}

torch::Tensor BaseVideoProcessor::temporalPad(const torch::Tensor& video,
                                              int temporal_patch_size) const {
  validateVideoTensor(video);

  if (temporal_patch_size <= 0) {
    LOG(FATAL) << "temporal_patch_size must be positive";
  }
  if (video.size(0) == 0) {
    return video;
  }

  auto pad_t = (temporal_patch_size - (video.size(0) % temporal_patch_size)) %
               temporal_patch_size;
  if (pad_t == 0) {
    return video;
  }

  auto last =
      video.index({video.size(0) - 1}).unsqueeze(0).repeat({pad_t, 1, 1, 1});
  return torch::cat({video, last}, 0);
}

}  // namespace xllm
