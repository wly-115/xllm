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

#include "core/framework/multimodal/mm_input_visitor.h"

#include <glog/logging.h>

#include <utility>

#include "processors/image_processor.h"

namespace xllm {

MMInputGatherVisitor::MMInputGatherVisitor(
    const ImageProcessor* image_processor,
    MMData& data)
    : image_processor_(image_processor), data_(data) {
  CHECK(image_processor_ != nullptr);
}

bool MMInputGatherVisitor::visit(const MMInputItem& item) {
  if (item.has_type(MMType::IMAGE)) {
    if (item.is_embedding()) {
      MMDataItem& data_item = data_.add(MMType::IMAGE);
      data_item.set_data(image_processor_->process_embedding(item.embedding));
    } else {
      images_.push_back(item.decode_image);
      data_.add(MMType::IMAGE);
    }
  }
  if (item.has_type(MMType::VIDEO)) {
    videos_.push_back(item.decode_video);
    video_metadata_.push_back(item.video_meta);
    data_.add(MMType::VIDEO);
  }
  if (item.has_type(MMType::AUDIO)) {
    audios_.push_back(item.decode_audio);
    audio_metadata_.push_back(item.audio_meta);
    data_.add(MMType::AUDIO);
  }
  return true;
}

}  // namespace xllm
