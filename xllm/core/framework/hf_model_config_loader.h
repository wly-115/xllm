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

#include <string>

#include "core/framework/model/model_args.h"
#include "core/framework/tokenizer/tokenizer_args.h"

namespace xllm {

class HFModelConfigLoader final {
 public:
  static bool load_model_args(const std::string& model_dir, ModelArgs* args);

  static bool load_tokenizer_args(const std::string& tokenizer_dir,
                                  const ModelArgs& model_args,
                                  TokenizerArgs* tokenizer_args);

  static bool load_image_preprocessor_args(const std::string& processor_dir,
                                           ModelArgs* args);

  static bool load_video_preprocessor_args(const std::string& processor_dir,
                                           ModelArgs* args);
};

}  // namespace xllm
