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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/message.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "processors/core/multimodal_input_processor.h"
#include "processors/core/prompt_processor.h"

namespace xllm {

class JinjaChatTemplate;
class Tokenizer;

struct PreprocessOutput {
  std::string prompt;
  std::vector<int> prompt_tokens;
  MMData mm_data;
};

class MultimodalProcessor;
std::unique_ptr<MultimodalProcessor> CreateMultimodalProcessor(
    std::unique_ptr<MultimodalInputProcessor> mm_processor,
    std::unique_ptr<PromptProcessor> prompt_processor,
    const TokenizerArgs& tokenizer_args,
    std::unique_ptr<Tokenizer> tokenizer);

class MultimodalProcessor {
 public:
  MultimodalProcessor(std::unique_ptr<MultimodalInputProcessor> mm_processor,
                      std::unique_ptr<PromptProcessor> prompt_processor,
                      std::unique_ptr<JinjaChatTemplate> chat_template,
                      std::unique_ptr<Tokenizer> tokenizer);
  ~MultimodalProcessor();

  bool preprocess(const std::vector<Message>& messages,
                  const std::string& payload,
                  PreprocessOutput& output);
  bool preprocess(const std::string& prompt,
                  MMData mm_data,
                  PreprocessOutput& output);

  bool encode_prompt(const std::string_view& prompt,
                     std::vector<int>& ids) const;

  bool process_mm_input(const MMInput& inputs, MMData& data);

 private:
  std::unique_ptr<MultimodalInputProcessor> mm_processor_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
  std::unique_ptr<JinjaChatTemplate> chat_template_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace xllm
