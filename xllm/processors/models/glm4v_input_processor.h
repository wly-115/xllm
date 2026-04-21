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
#include "processors/models/glm4v_image_processor.h"
#include "processors/models/glm4v_video_processor.h"

namespace xllm {

class Glm4VInputProcessor : public MultimodalInputProcessor {
 public:
  Glm4VInputProcessor(const ModelArgs&);
  ~Glm4VInputProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

 private:
  bool process_images(const std::vector<torch::Tensor>& images,
                      std::vector<MMDataItem>& image_items);
  bool process_videos(const std::vector<torch::Tensor>& videos,
                      const std::vector<VideoMetadata>& video_meta_list,
                      std::vector<MMDataItem>& video_items);

 private:
  Glm4VImageProcessor image_processor_;
  Glm4VVideoProcessor video_processor_;
};

}  // namespace xllm
