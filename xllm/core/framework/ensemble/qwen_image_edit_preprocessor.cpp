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

#include "core/framework/ensemble/qwen_image_edit_preprocessor.h"

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/ensemble/node_payload_fields.h"
#include "core/framework/hf_model_config_loader.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/multimodal/mm_type.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/framework/tokenizer/tokenizer_factory.h"
#include "processors/multimodal_processor.h"
#include "processors/transforms.h"

namespace xllm {
namespace {

constexpr char kQwenImageEditType[] = "qwen-image-edit";
constexpr int64_t kConditionImageArea = 384 * 384;
constexpr int64_t kImageSizeAlignment = 32;
constexpr int32_t kBicubicResample = 3;
constexpr char kPromptPrefix[] =
    "<|im_start|>system\nDescribe the key features of the input image "
    "(color, shape, size, texture, objects, background), then explain how "
    "the user's text instruction should alter or modify the image. Generate "
    "a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\nPicture 1: "
    "<|vision_start|><|image_pad|><|vision_end|>";
constexpr char kPromptSuffix[] = "<|im_end|>\n<|im_start|>assistant\n";

MMInput build_mm_input(const torch::Tensor& image) {
  MMInputItem item;
  item.type = MMType::IMAGE;
  item.decode_image = image.cpu().contiguous();

  MMInput mm_input;
  mm_input.insert({item});
  return mm_input;
}

torch::Tensor vae_resize(const torch::Tensor& image) {
  CHECK_EQ(image.dim(), 3)
      << "Qwen image edit input image must have shape [C, H, W].";
  const int64_t image_height = image.size(1);
  const int64_t image_width = image.size(2);

  const double aspect_ratio =
      static_cast<double>(image_width) / static_cast<double>(image_height);
  const double resized_width =
      std::sqrt(static_cast<double>(kConditionImageArea) * aspect_ratio);
  const double resized_height = resized_width / aspect_ratio;
  const int64_t target_width =
      static_cast<int64_t>(std::round(
          resized_width / static_cast<double>(kImageSizeAlignment))) *
      kImageSizeAlignment;
  const int64_t target_height =
      static_cast<int64_t>(std::round(
          resized_height / static_cast<double>(kImageSizeAlignment))) *
      kImageSizeAlignment;

  return transforms::resize(
      image, {target_height, target_width}, kBicubicResample);
}

}  // namespace

QwenImageEditPreprocessor::QwenImageEditPreprocessor(
    std::unique_ptr<MultimodalProcessorBase> multimodal_processor)
    : multimodal_processor_(std::move(multimodal_processor)) {}

QwenImageEditPreprocessor::~QwenImageEditPreprocessor() = default;

NodePayload QwenImageEditPreprocessor::preprocess(
    const OmniRequestInput& input) const {
  CHECK(!input.prompt.empty()) << "prompt cannot be empty.";
  CHECK(input.image.defined()) << "image tensor must be defined.";

  torch::Tensor encode_image = vae_resize(input.image);
  MMInput mm_input = build_mm_input(encode_image);
  MMData mm_data;
  CHECK(multimodal_processor_->process_multimodal(mm_input, mm_data))
      << "Failed to process image input.";

  std::string prompt =
      std::string(kPromptPrefix) + input.prompt + kPromptSuffix;
  MMData prompt_mm_data = mm_data;
  std::vector<int32_t> prompt_tokens;
  CHECK(multimodal_processor_->process_prompt(
      prompt, prompt_mm_data, prompt_tokens))
      << "Failed to preprocess prompt.";

  std::string negative_prompt =
      std::string(kPromptPrefix) + input.negative_prompt + kPromptSuffix;
  MMData negative_prompt_mm_data = mm_data;
  std::vector<int32_t> negative_prompt_tokens;
  CHECK(multimodal_processor_->process_prompt(
      negative_prompt, negative_prompt_mm_data, negative_prompt_tokens))
      << "Failed to preprocess negative_prompt.";

  NodePayload payload;
  CHECK(payload.set(kPayloadFieldPrompt, prompt));
  CHECK(payload.set(kPayloadFieldPromptTokens, prompt_tokens));
  CHECK(payload.set(kPayloadFieldPromptMmData, prompt_mm_data));
  CHECK(payload.set(kPayloadFieldNegativePrompt, negative_prompt));
  CHECK(payload.set(kPayloadFieldNegativePromptTokens, negative_prompt_tokens));
  CHECK(
      payload.set(kPayloadFieldNegativePromptMmData, negative_prompt_mm_data));
  CHECK(payload.set(kPayloadFieldImage, input.image.cpu().contiguous()));
  return payload;
}

std::unique_ptr<QwenImageEditPreprocessor> create_qwen_image_edit_preprocessor(
    const PreprocessorConfig& config) {
  CHECK_EQ(config.type, kQwenImageEditType) << "Unsupported preprocessor type.";
  CHECK(!config.model_root.empty())
      << "preprocessor model_root cannot be empty.";

  const std::filesystem::path model_root(config.model_root);
  const std::string text_encoder_dir = (model_root / "text_encoder").string();
  const std::string processor_dir = (model_root / "processor").string();

  ModelArgs model_args;
  CHECK(HFModelConfigLoader::load_model_args(text_encoder_dir, &model_args))
      << "Failed to load text encoder model args from " << text_encoder_dir;

  TokenizerArgs tokenizer_args;
  CHECK(HFModelConfigLoader::load_tokenizer_args(
      text_encoder_dir, model_args, &tokenizer_args))
      << "Failed to load tokenizer args from " << text_encoder_dir;

  CHECK(HFModelConfigLoader::load_image_preprocessor_args(processor_dir,
                                                          &model_args))
      << "Failed to load image preprocessor args from " << processor_dir;
  CHECK(HFModelConfigLoader::load_video_preprocessor_args(processor_dir,
                                                          &model_args))
      << "Failed to load video preprocessor args from " << processor_dir;

  std::unique_ptr<Tokenizer> tokenizer =
      TokenizerFactory::create_tokenizer(text_encoder_dir, tokenizer_args);
  CHECK(tokenizer != nullptr)
      << "Failed to create tokenizer from " << text_encoder_dir;

  std::shared_ptr<Tokenizer> shared_tokenizer(std::move(tokenizer));
  std::unique_ptr<MultimodalProcessorBase> multimodal_processor =
      create_multimodal_processor(model_args, std::move(shared_tokenizer));
  CHECK(multimodal_processor != nullptr)
      << "Failed to create multimodal processor for "
      << model_args.model_type();

  return std::make_unique<QwenImageEditPreprocessor>(
      std::move(multimodal_processor));
}

}  // namespace xllm
