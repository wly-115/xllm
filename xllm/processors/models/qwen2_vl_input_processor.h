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

#include <vector>

#include "processors/core/multimodal_input_processor.h"
#include "processors/models/qwen2_vl_image_processor.h"
#include "processors/models/qwen2_vl_video_processor.h"

namespace xllm {

class Qwen2VLInputProcessor : public MultimodalInputProcessor {
 public:
  Qwen2VLInputProcessor(const ModelArgs&);
  ~Qwen2VLInputProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_images_embedding(
      const std::vector<EmbeddingOutput>& images_embedding,
      MMData& mm_datas);
  bool process_videos(std::vector<torch::Tensor> videos,
                      std::vector<VideoMetadata> video_meta_list,
                      MMData& mm_datas);

 private:
  Qwen2VLImageProcessor image_processor_;
  Qwen2VLVideoProcessor video_processor_;
};

}  // namespace xllm
