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

#include "processors/models/clip_input_processor.h"

namespace xllm {

CLIPInputProcessor::CLIPInputProcessor(const ModelArgs& args)
    : image_processor_(args) {}

bool CLIPInputProcessor::process(const MMInput& mm_inputs, MMData& mm_datas) {
  return false;
}

torch::Tensor CLIPInputProcessor::process_images(const torch::Tensor& images) {
  return image_processor_.process_images(images);
}

}  // namespace xllm
