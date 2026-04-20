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

#include "processors/core/multimodal_processor.h"

#include <utility>

#include "common/metrics.h"
#include "core/framework/chat_template/jinja_chat_template.h"
#include "core/framework/request/mm_input.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "util/timer.h"

namespace xllm {

std::unique_ptr<MultimodalProcessor> CreateMultimodalProcessor(
    std::unique_ptr<MultimodalInputProcessor> mm_processor,
    std::unique_ptr<PromptProcessor> prompt_processor,
    const TokenizerArgs& tokenizer_args,
    std::unique_ptr<Tokenizer> tokenizer) {
  return std::make_unique<MultimodalProcessor>(
      std::move(mm_processor),
      std::move(prompt_processor),
      std::make_unique<JinjaChatTemplate>(tokenizer_args),
      std::move(tokenizer));
}

MultimodalProcessor::MultimodalProcessor(
    std::unique_ptr<MultimodalInputProcessor> mm_processor,
    std::unique_ptr<PromptProcessor> prompt_processor,
    std::unique_ptr<JinjaChatTemplate> chat_template,
    std::unique_ptr<Tokenizer> tokenizer)
    : mm_processor_(std::move(mm_processor)),
      prompt_processor_(std::move(prompt_processor)),
      chat_template_(std::move(chat_template)),
      tokenizer_(std::move(tokenizer)) {}

MultimodalProcessor::~MultimodalProcessor() = default;

bool MultimodalProcessor::encode_prompt(const std::string_view& prompt,
                                        std::vector<int>& ids) const {
  return tokenizer_->encode(prompt, &ids);
}

bool MultimodalProcessor::process_mm_input(const MMInput& inputs,
                                           MMData& data) {
  return mm_processor_->process(inputs, data);
}

bool MultimodalProcessor::preprocess(const std::vector<Message>& messages,
                                     const std::string& payload,
                                     PreprocessOutput& output) {
  static MMInputTransfer mm_input_transfer;

  MMInput mm_inputs(payload);
  MMErrCode code = mm_input_transfer.trans(messages, mm_inputs);
  if (code != MMErrCode::SUCCESS) {
    LOG(ERROR) << MMErrToString(code);
    return false;
  }

  if (!mm_inputs.empty() && !process_mm_input(mm_inputs, output.mm_data)) {
    LOG(ERROR) << "multimodal processor process failed.";
    return false;
  }

  Timer timer;
  auto rendered_prompt = chat_template_->apply(messages);
  if (!rendered_prompt.has_value()) {
    LOG(ERROR) << "Failed to construct prompt from messages";
    return false;
  }
  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  output.prompt = std::move(rendered_prompt.value());
  prompt_processor_->process(output.prompt, output.mm_data);
  Timer encode_timer;
  if (!encode_prompt(output.prompt, output.prompt_tokens)) {
    LOG(ERROR) << "Failed to encode prompt: " << output.prompt;
    return false;
  }
  prompt_processor_->find_mm_spans(output.prompt_tokens, output.mm_data);
  COUNTER_ADD(tokenization_latency_seconds, encode_timer.elapsed_seconds());
  return true;
}

bool MultimodalProcessor::preprocess(const std::string& prompt,
                                     MMData mm_data,
                                     PreprocessOutput& output) {
  output.prompt = prompt;
  output.mm_data = std::move(mm_data);
  prompt_processor_->process(output.prompt, output.mm_data);
  Timer timer;
  if (!encode_prompt(output.prompt, output.prompt_tokens)) {
    LOG(ERROR) << "Failed to encode prompt: " << output.prompt;
    return false;
  }
  prompt_processor_->find_mm_spans(output.prompt_tokens, output.mm_data);
  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());
  return true;
}

}  // namespace xllm
