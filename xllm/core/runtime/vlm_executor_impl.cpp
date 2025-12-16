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

#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "util/multimodal_util.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  LOG(INFO) << "VlmExecutorImpl constructor";
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(options_.num_decoding_tokens(), 0, args_);
}

std::vector<torch::Tensor> VlmExecutorImpl::execute_encoder(
    const torch::Tensor& inputs,
    const ModelInputParams& params) {
  return dynamic_cast<CausalVLM*>(model_)->execute_encoder(inputs, params);
}

torch::Tensor VlmExecutorImpl::run(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
  torch::NoGradGuard no_grad;

  auto multimodal_embeds = execute_encoder(tokens, params);
  auto& mm_datas = params.mm_data.get_mm_datas();

  update_multimodal_embeddings(mm_datas, params.mm_data, multimodal_embeds);
  auto visual_features = gather_multimodal_embeddings(mm_datas);

  params.input_embedding =
      dynamic_cast<CausalVLM*>(model_)->merge_multimodal_embeddings(
          tokens, visual_features);

  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm
