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

#include "processors/multimodal_processor.h"

#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/util/hash_util.h"
#include "models/model_registry.h"
#include "util/timer.h"

namespace xllm {

MultimodalProcessorBase::MultimodalProcessorBase(
    const TokenizerArgs& tokenizer_args,
    std::shared_ptr<Tokenizer> tokenizer)
    : chat_template_(std::make_unique<JinjaChatTemplate>(tokenizer_args)),
      tokenizer_(std::move(tokenizer)) {}

MultimodalProcessorBase::~MultimodalProcessorBase() = default;

bool MultimodalProcessorBase::apply_chat_template(
    const std::vector<Message>& messages,
    const std::vector<JsonTool>& tools,
    const nlohmann::ordered_json& chat_template_kwargs,
    std::string& prompt,
    std::string& error_message) const {
  Timer timer;
  std::optional<std::string> rendered_prompt =
      chat_template_->apply(messages, tools, chat_template_kwargs);
  if (!rendered_prompt.has_value()) {
    error_message = "Failed to construct prompt from messages";
    LOG(ERROR) << error_message;
    return false;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  prompt = std::move(rendered_prompt.value());
  return true;
}

bool MultimodalProcessorBase::tokenize(const std::string& prompt,
                                       std::vector<int32_t>& token_ids,
                                       std::string& error_message) const {
  Timer timer;
  if (!tokenizer_->encode(prompt, &token_ids)) {
    error_message = "Failed to encode prompt: " + prompt;
    LOG(ERROR) << error_message;
    return false;
  }
  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());
  return true;
}

void MultimodalProcessorBase::hash_mm_items(const MMInput& mm_input,
                                            MMData& mm_data) {
  const auto& mm_input_items = mm_input.items();
  auto& mm_items = mm_data.items<MMItemVec>();
  size_t size = mm_input_items.size();
  for (size_t idx = 0; idx < size; ++idx) {
    const std::string& data = mm_input_items[idx].raw_data;
    if (!data.empty()) {
      XXH3Key mm_hash = hash_string(data);
      auto& schedule_data =
          mm_items[idx].mutable_state().mutable_schedule_data();
      schedule_data.key = mm_hash;
    } else {
      LOG(WARNING) << "Empty data for multimodal item";
    }
  }
}

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    const TokenizerArgs& tokenizer_args,
    std::shared_ptr<Tokenizer> tokenizer) {
  const std::string& model_type = model_args.model_type();
  std::string resolved_name;
  std::string error_message;
  CHECK(resolve_model_registration_name(
      model_type, &resolved_name, &error_message))
      << error_message;

  MultimodalProcessorFactory multimodal_processor_factory =
      ModelRegistry::get_multimodal_processor_factory(resolved_name);
  CHECK(multimodal_processor_factory != nullptr)
      << "Missing multimodal processor for model type: " << model_type;
  return multimodal_processor_factory(
      model_args, tokenizer_args, std::move(tokenizer));
}

}  // namespace xllm
