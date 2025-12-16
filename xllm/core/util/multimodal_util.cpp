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
#include "multimodal_util.h"

#include "core/framework/request/mm_batch_data.h"
#include "core/framework/request/mm_input.h"

namespace xllm {
void update_multimodal_embeddings(
    std::vector<MMData>& mm_datas,
    const MMBatchData& batch_mm_data,
    std::vector<torch::Tensor>& multimodal_embeddings) {
  int32_t image_num = 0;
  if (const auto& res = batch_mm_data.get<torch::Tensor>("image_grid_thw")) {
    image_num = res.value().size(0);
  }
  int32_t video_num = 0;
  if (const auto& res = batch_mm_data.get<torch::Tensor>("video_grid_thw")) {
    video_num = res.value().size(0);
  }
  // TODO:audio
  auto image_begin = multimodal_embeddings.begin();
  auto image_end = image_begin + image_num;
  std::vector<torch::Tensor> image_emb(image_begin, image_end);

  std::vector<torch::Tensor> video_emb(image_end, multimodal_embeddings.end());
  int32_t image_idx = 0;
  int32_t video_idx = 0;

  for (auto& mm_data : mm_datas) {
    int32_t mm_num = mm_data.size();
    auto& items = mm_data.items<MMItemVec>();
    for (int idx = 0; idx < mm_num; ++idx) {
      auto& mm_item = items[idx];
      int cached_tokens_num = mm_item.get_cached_tokens_num();
      auto mm_position = mm_item.get_mm_position();
      int tokens_per_modality = mm_position.length;
      if (tokens_per_modality == cached_tokens_num) continue;
      auto mask =
          torch::ones({tokens_per_modality}, torch::dtype(torch::kBool));
      mask.index({torch::indexing::Slice(0, cached_tokens_num)}) = false;
      std::optional<torch::Tensor> embedding =
          mm_item.get<torch::Tensor>("embedding");
      torch::Tensor updated_embedding;
      auto mm_type = mm_item.type();

      switch (mm_type) {
        case MMType::IMAGE:
          updated_embedding =
              embedding ? embedding.value() : image_emb[image_idx++];
          break;
        case MMType::VIDEO:

          updated_embedding =
              embedding ? embedding.value() : video_emb[video_idx++];
          break;
        default:
          LOG(FATAL) << "wrong MMType.";
          break;
      }
      if (updated_embedding.defined()) {
        mm_item.update<torch::Tensor>("embedding",
                                      updated_embedding.index({mask}));
      }
    }
  }
}

torch::Tensor gather_multimodal_embeddings(std::vector<MMData>& mm_datas) {
  std::vector<torch::Tensor> multimodal_embeddings;
  for (auto& mm_data : mm_datas) {
    int32_t mm_num = mm_data.size();
    auto& mm_items = mm_data.items<MMItemVec>();
    for (int idx = 0; idx < mm_num; ++idx) {
      auto& mm_item = mm_items[idx];
      int cached_tokens_num = mm_item.get_cached_tokens_num();
      int tokens_per_modality = mm_item.get_mm_position().length;
      if (tokens_per_modality == cached_tokens_num) continue;
      multimodal_embeddings.push_back(
          mm_item.get<torch::Tensor>("embedding").value());
    }
  }
  if (multimodal_embeddings.empty()) {
    return torch::Tensor();
  }
  return torch::cat(multimodal_embeddings, 0);
}
}  // namespace xllm