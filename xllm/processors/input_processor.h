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

#include <torch/torch.h>

#include <string>

#include "core/framework/request/mm_data.h"

namespace xllm {
class ImageProcessor;
class VideoProcessor;

class InputProcessor {
 public:
  virtual ~InputProcessor() = default;

  virtual bool process(std::string& prompt,
                       const MMInput& mm_inputs,
                       MMData& mm_data) = 0;
  virtual bool process_multimodal_inputs(const MMInput& mm_inputs,
                                         MMData& mm_data) = 0;
  virtual void replace_placeholder(std::string& prompt,
                                   const MMData& mm_data) = 0;

 private:
  std::unique_ptr<ImageProcessor> image_processor_;
  std::unique_ptr<VideoProcessor> video_processor_;
};

}  // namespace xllm
