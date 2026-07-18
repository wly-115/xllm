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

#include "core/distributed_runtime/qwen_vlm_encode_adapter.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/ensemble/node_payload_fields.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_state.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace {

constexpr char kPromptSuffix[] = ":prompt";
constexpr char kNegativePromptSuffix[] = ":negative_prompt";
constexpr char kOutputFieldMetadataKey[] = "omni.output_field";
constexpr int64_t kPromptPrefixTokenCount = 64;

std::shared_ptr<Request> make_embedding_request(
    const std::string& request_id,
    const std::string& prompt,
    const std::vector<int32_t>& tokens,
    MMData mm_data,
    const ModelArgs& model_args,
    const std::string& output_field,
    RequestSamplingParam sampling_param) {
  sampling_param.is_embeddings = true;

  StoppingChecker stopping_checker(
      /*max_tokens=*/1,
      model_args.max_position_embeddings(),
      model_args.eos_token_id(),
      /*ignore_eos=*/false,
      model_args.stop_token_ids(),
      {});

  const size_t capacity = tokens.size() + 1;
  RequestState state(prompt,
                     tokens,
                     std::move(mm_data),
                     sampling_param,
                     stopping_checker,
                     capacity,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     nullptr,
                     nullptr);
  state.metadata.emplace(kOutputFieldMetadataKey, output_field);
  return std::make_shared<Request>(request_id, "", "", std::move(state));
}

}  // namespace

QwenVlmEncodeAdapter::QwenVlmEncodeAdapter(const ModelArgs& model_args)
    : model_args_(model_args) {}

std::vector<std::shared_ptr<Request>> QwenVlmEncodeAdapter::build_requests(
    const NodeData& input) {
  CHECK(!input.context.request_id.empty()) << "Request id cannot be empty.";

  auto prompt = input.payload.get<std::string>(kPayloadFieldPrompt);
  auto prompt_tokens =
      input.payload.get<std::vector<int32_t>>(kPayloadFieldPromptTokens);
  auto prompt_mm_data = input.payload.get<MMData>(kPayloadFieldPromptMmData);
  CHECK(prompt.has_value()) << "Missing prompt field in payload.";
  CHECK(prompt_tokens.has_value()) << "Missing prompt tokens in payload.";
  CHECK(prompt_mm_data.has_value()) << "Missing prompt MMData in payload.";

  auto neg_prompt = input.payload.get<std::string>(kPayloadFieldNegativePrompt);
  auto neg_tokens = input.payload.get<std::vector<int32_t>>(
      kPayloadFieldNegativePromptTokens);
  auto neg_mm_data =
      input.payload.get<MMData>(kPayloadFieldNegativePromptMmData);
  CHECK(neg_prompt.has_value()) << "Missing negative prompt field in payload.";
  CHECK(neg_tokens.has_value()) << "Missing negative prompt tokens in payload.";
  CHECK(neg_mm_data.has_value())
      << "Missing negative prompt MMData in payload.";

  RequestSamplingParam sampling_params =
      input.context.sampling_params.value_or(RequestSamplingParam{});
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(2);
  requests.emplace_back(
      make_embedding_request(input.context.request_id + kPromptSuffix,
                             *prompt,
                             *prompt_tokens,
                             std::move(*prompt_mm_data),
                             model_args_,
                             kPayloadFieldPromptEmbed,
                             sampling_params));
  requests.emplace_back(
      make_embedding_request(input.context.request_id + kNegativePromptSuffix,
                             *neg_prompt,
                             *neg_tokens,
                             std::move(*neg_mm_data),
                             model_args_,
                             kPayloadFieldNegativePromptEmbed,
                             sampling_params));
  return requests;
}

NodeData QwenVlmEncodeAdapter::convert_output(
    const NodeData& input,
    const std::vector<RequestOutput>& outputs) {
  CHECK_EQ(outputs.size(), 2) << "Expected 2 VLM embedding outputs.";

  auto extract_embedding = [](const RequestOutput& output) -> torch::Tensor {
    CHECK(!output.status.has_value() || output.status->ok())
        << "VLM request failed: " << output.status->message();
    CHECK(!output.outputs.empty()) << "VLM output cannot be empty.";
    CHECK(output.outputs[0].mm_embeddings.has_value())
        << "VLM output does not contain multimodal embeddings.";
    CHECK(!output.outputs[0].mm_embeddings->empty())
        << "VLM multimodal embeddings cannot be empty.";
    const torch::Tensor& embedding =
        output.outputs[0].mm_embeddings.value().front().embedding;
    CHECK(embedding.defined()) << "VLM embedding tensor is undefined.";
    CHECK_EQ(embedding.dim(), 2)
        << "Qwen image edit VLM embedding must be 2-dimensional.";
    CHECK_GT(embedding.size(0), kPromptPrefixTokenCount)
        << "Qwen image edit VLM embedding is shorter than prompt prefix.";
    return embedding.slice(/*dim=*/0,
                           /*start=*/kPromptPrefixTokenCount);
  };

  std::optional<torch::Tensor> prompt_embed;
  std::optional<torch::Tensor> negative_prompt_embed;
  for (const RequestOutput& output : outputs) {
    auto metadata_it = output.metadata.find(kOutputFieldMetadataKey);
    CHECK(metadata_it != output.metadata.end())
        << "VLM output is missing omni.output_field metadata.";

    torch::Tensor embedding = extract_embedding(output);
    if (metadata_it->second == kPayloadFieldPromptEmbed) {
      CHECK(!prompt_embed.has_value()) << "Duplicated prompt embedding output.";
      prompt_embed = embedding;
      continue;
    }
    if (metadata_it->second == kPayloadFieldNegativePromptEmbed) {
      CHECK(!negative_prompt_embed.has_value())
          << "Duplicated negative prompt embedding output.";
      negative_prompt_embed = embedding;
      continue;
    }
    LOG(FATAL) << "Unsupported VLM output field metadata: "
               << metadata_it->second;
  }
  CHECK(prompt_embed.has_value()) << "Missing prompt embedding output field.";
  CHECK(negative_prompt_embed.has_value())
      << "Missing negative prompt embedding output field.";

  auto image = input.payload.get<torch::Tensor>(kPayloadFieldImage);
  CHECK(image.has_value()) << "Missing image field in input payload.";

  NodePayload payload;
  CHECK(payload.set(kPayloadFieldPromptEmbed, *prompt_embed));
  CHECK(payload.set(kPayloadFieldNegativePromptEmbed, *negative_prompt_embed));
  CHECK(payload.set(kPayloadFieldImage, *image));
  auto condition_image =
      input.payload.get<torch::Tensor>(kPayloadFieldConditionImage);
  if (condition_image.has_value()) {
    CHECK(payload.set(kPayloadFieldConditionImage, *condition_image));
  }

  NodeData output;
  output.context = input.context;
  output.payload = std::move(payload);
  return output;
}

}  // namespace xllm
