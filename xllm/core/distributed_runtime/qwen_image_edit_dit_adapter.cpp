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

#include "core/distributed_runtime/qwen_image_edit_dit_adapter.h"

#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/ensemble/node_payload_fields.h"
#include "core/framework/request/dit_request.h"
#include "core/framework/request/dit_request_output.h"
#include "core/framework/request/dit_request_state.h"

namespace xllm {
namespace {

constexpr char kVlmEmbeddingPrompt[] = "<omni-vlm-embedding>";
constexpr char kVlmEmbeddingNegativePrompt[] = "<omni-negative-vlm-embedding>";

}  // namespace

std::vector<std::shared_ptr<DiTRequest>>
QwenImageEditDitAdapter::build_requests(const NodeData& input) {
  CHECK(!input.context.request_id.empty()) << "Request id cannot be empty.";

  auto prompt_embed =
      input.payload.get<torch::Tensor>(kPayloadFieldPromptEmbed);
  auto neg_embed =
      input.payload.get<torch::Tensor>(kPayloadFieldNegativePromptEmbed);
  auto image = input.payload.get<torch::Tensor>(kPayloadFieldImage);
  CHECK(prompt_embed.has_value()) << "Missing prompt embedding in payload.";
  CHECK(neg_embed.has_value())
      << "Missing negative prompt embedding in payload.";
  CHECK(image.has_value()) << "Missing image in payload.";
  CHECK(input.context.dit_generation_params.has_value())
      << "Request context is missing DiT generation params.";
  DiTGenerationParams generation_params = *input.context.dit_generation_params;

  DiTInputParams input_params;
  input_params.prompt = kVlmEmbeddingPrompt;
  input_params.negative_prompt = kVlmEmbeddingNegativePrompt;
  input_params.prompt_embed = std::move(*prompt_embed);
  input_params.negative_prompt_embed = std::move(*neg_embed);
  input_params.image = std::move(*image);

  DiTRequestState state(input_params, generation_params, nullptr, nullptr);
  return {std::make_shared<DiTRequest>(
      input.context.request_id, "", "", std::move(state))};
}

NodeData QwenImageEditDitAdapter::convert_output(
    const NodeData& input,
    const std::vector<DiTRequestOutput>& outputs) {
  CHECK_EQ(outputs.size(), 1) << "Expected exactly one DiT output.";
  const DiTRequestOutput& output = outputs[0];
  CHECK(!output.status.has_value() || output.status->ok())
      << "DiT request failed: " << output.status->message();

  NodePayload payload;
  if (!output.outputs.empty()) {
    const DiTGenerationOutput& gen_output = output.outputs[0];
    CHECK(payload.set(kPayloadFieldWidth, std::to_string(gen_output.width)));
    CHECK(payload.set(kPayloadFieldHeight, std::to_string(gen_output.height)));
    CHECK(payload.set(kPayloadFieldSeed, std::to_string(gen_output.seed)));
  }
  for (size_t i = 0; i < output.outputs.size(); ++i) {
    std::string field =
        std::string(kPayloadFieldImage) + "_" + std::to_string(i);
    const std::string& image = output.outputs[i].image;
    NodePayloadBytes image_bytes(image.begin(), image.end());
    CHECK(payload.set(field, std::move(image_bytes)));
  }

  NodeData node_output;
  node_output.context = input.context;
  node_output.payload = std::move(payload);
  return node_output;
}

}  // namespace xllm
