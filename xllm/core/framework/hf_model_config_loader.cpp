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

#include "core/framework/hf_model_config_loader.h"

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"
#include "core/util/utils.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

JsonReader normalize_config_torch_dtype(const JsonReader& reader) {
  nlohmann::json config = reader.data();
  if (!config.contains("torch_dtype") && config.contains("dtype")) {
    config["torch_dtype"] = config["dtype"];
  }

  JsonReader normalized_reader;
  normalized_reader.parse_text(config.dump());
  return normalized_reader;
}

std::optional<std::string> load_chat_template_file(const std::string& dir) {
  const std::string chat_template_path = dir + "/chat_template.json";
  JsonReader reader;
  if (reader.parse(chat_template_path);
      std::optional<std::string> value =
          reader.value<std::string>("chat_template")) {
    return value;
  }

  const std::string raw_chat_template_path = dir + "/chat_template.jinja";
  std::ifstream file(raw_chat_template_path);
  if (file.is_open()) {
    std::ostringstream content;
    content << file.rdbuf();
    file.close();
    return content.str();
  }
  return std::nullopt;
}

}  // namespace

bool HFModelConfigLoader::load_model_args(const std::string& model_dir,
                                          ModelArgs* args) {
  if (args == nullptr) {
    LOG(ERROR) << "ModelArgs output cannot be null.";
    return false;
  }

  JsonReader reader;
  const std::string args_file_path = model_dir + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  const std::string model_type =
      util::get_model_type(reader,
                           std::filesystem::path(model_dir),
                           ModelConfig::get_instance().backend());

  std::string resolved_model_type;
  std::string error_message;
  if (!resolve_model_registration_name(
          model_type, &resolved_model_type, &error_message)) {
    LOG(ERROR) << error_message;
    return false;
  }

  ModelArgsLoader model_args_loader =
      ModelRegistry::get_model_args_loader(resolved_model_type);
  if (model_args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << resolved_model_type;
    return false;
  }
  const JsonReader config_reader = normalize_config_torch_dtype(reader);
  model_args_loader(config_reader, args);
  args->enable_mla(util::should_enable_mla(
      std::filesystem::path(model_dir), ModelConfig::get_instance().backend()));

  return true;
}

bool HFModelConfigLoader::load_tokenizer_args(const std::string& tokenizer_dir,
                                              const ModelArgs& model_args,
                                              TokenizerArgs* tokenizer_args) {
  if (tokenizer_args == nullptr) {
    LOG(ERROR) << "TokenizerArgs output cannot be null.";
    return false;
  }

  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      tokenizer_dir + "/tokenizer_config.json";

  const std::string tokenizer_json_path = tokenizer_dir + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path)) {
    tokenizer_args->tokenizer_type() = "fast";
    tokenizer_args->vocab_file() = tokenizer_json_path;
  }

  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    if (std::optional<std::string> value =
            load_chat_template_file(tokenizer_dir)) {
      tokenizer_args->chat_template() = value.value();
    } else if (std::optional<std::string> value =
                   tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args->chat_template() = value.value();
    }
    if (std::optional<bool> value =
            tokenizer_reader.value<bool>("add_bos_token")) {
      tokenizer_args->add_bos_token() = value.value();
    }
    if (std::optional<bool> value =
            tokenizer_reader.value<bool>("add_eos_token")) {
      tokenizer_args->add_eos_token() = value.value();
    }
    if (std::optional<std::string> value =
            tokenizer_reader.value<std::string>("tokenizer_class")) {
      tokenizer_args->tokenizer_class() = value.value();
    }
    if (std::optional<std::string> value =
            tokenizer_reader.value<std::string>("bos_token.content")) {
      tokenizer_args->bos_token() = value.value();
    } else if (std::optional<std::string> value =
                   tokenizer_reader.value<std::string>("bos_token")) {
      tokenizer_args->bos_token() = value.value();
    }
    if (std::optional<std::string> value =
            tokenizer_reader.value<std::string>("eos_token.content")) {
      tokenizer_args->eos_token() = value.value();
    } else if (std::optional<std::string> value =
                   tokenizer_reader.value<std::string>("eos_token")) {
      tokenizer_args->eos_token() = value.value();
    }
    if (std::optional<std::string> value =
            tokenizer_reader.value<std::string>("pad_token.content")) {
      tokenizer_args->pad_token() = value.value();
    } else if (std::optional<std::string> value =
                   tokenizer_reader.value<std::string>("pad_token")) {
      tokenizer_args->pad_token() = value.value();
    }
  }

  TokenizerArgsLoader tokenizer_args_loader =
      ModelRegistry::get_tokenizer_args_loader(model_args.model_type());
  if (tokenizer_args_loader != nullptr &&
      !tokenizer_args_loader(tokenizer_reader, tokenizer_args)) {
    LOG(ERROR) << "Failed to load tokenizer args from "
               << tokenizer_args_file_path;
    return false;
  }

  return true;
}

bool HFModelConfigLoader::load_image_preprocessor_args(
    const std::string& processor_dir,
    ModelArgs* args) {
  if (args == nullptr) {
    LOG(ERROR) << "ModelArgs output cannot be null.";
    return false;
  }

  JsonReader image_preprocess_reader;
  const std::string image_preprocess_file_path =
      processor_dir + "/preprocessor_config.json";
  if (image_preprocess_reader.parse(image_preprocess_file_path)) {
    LOG(INFO) << "Success to parse image preprocess args file: "
              << image_preprocess_file_path;
    args->mm_image_do_center_crop() =
        image_preprocess_reader.value_or<bool>("do_center_crop", false);
    args->mm_image_crop_height_size() =
        image_preprocess_reader.value_or<int>("crop_size.height", 335);
    args->mm_image_crop_width_size() =
        image_preprocess_reader.value_or<int>("crop_size.width", 335);

    args->mm_image_do_resize() =
        image_preprocess_reader.value_or<bool>("do_resize", false);
    args->mm_image_resize_shortest_edge() =
        image_preprocess_reader.value_or<int>("size.shortest_edge", 335);
    args->mm_image_resample() =
        image_preprocess_reader.value_or<int>("resample", 335);

    args->mm_image_do_rescale() =
        image_preprocess_reader.value_or<bool>("do_rescale", false);
    args->mm_image_rescale_factor() =
        image_preprocess_reader.value_or<double>("rescale_factor", 0);

    args->mm_image_do_normalize() =
        image_preprocess_reader.value_or<bool>("do_normalize", false);

    const nlohmann::json& image_preprocess_data =
        image_preprocess_reader.data();
    if (image_preprocess_reader.contains("image_mean")) {
      args->mm_image_normalize_mean() =
          image_preprocess_data["image_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("image_std")) {
      args->mm_image_normalize_std() =
          image_preprocess_data["image_std"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_mean")) {
      args->mm_image_normalize_mean() =
          image_preprocess_data["norm_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_std")) {
      args->mm_image_normalize_std() =
          image_preprocess_data["norm_std"].get<std::vector<double>>();
    }

    args->mm_image_shortest_edge() =
        image_preprocess_reader.value_or<int>("size.shortest_edge", 0);
    args->mm_image_longest_edge() =
        image_preprocess_reader.value_or<int>("size.longest_edge", 0);
    args->mm_image_min_pixels() =
        image_preprocess_reader.value_or<int>("min_pixels", 0);
    args->mm_image_max_pixels() =
        image_preprocess_reader.value_or<int>("max_pixels", 0);
    args->mm_image_patch_size() =
        image_preprocess_reader.value_or<int>("patch_size", 0);
    args->mm_image_temporal_patch_size() =
        image_preprocess_reader.value_or<int>("temporal_patch_size", 0);
    args->mm_image_merge_size() =
        image_preprocess_reader.value_or<int>("merge_size", 0);
    args->mm_image_feature_size() =
        image_preprocess_reader.value_or<int>("image_feature_size", 0);
    args->mm_scale_resolution() =
        image_preprocess_reader.value_or<int>("scale_resolution", 0);
    args->mm_slice_mode() =
        image_preprocess_reader.value_or<bool>("slice_mode", false);
    args->mm_use_image_id() =
        image_preprocess_reader.value_or<bool>("use_image_id", false);
  }

  return true;
}

bool HFModelConfigLoader::load_video_preprocessor_args(
    const std::string& processor_dir,
    ModelArgs* args) {
  if (args == nullptr) {
    LOG(ERROR) << "ModelArgs output cannot be null.";
    return false;
  }

  JsonReader video_preprocess_reader;
  const std::string video_preprocess_file_path =
      processor_dir + "/video_preprocessor_config.json";
  if (video_preprocess_reader.parse(video_preprocess_file_path)) {
    LOG(INFO) << "Success to parse video preprocess args file: "
              << video_preprocess_file_path;

    args->mm_video_shortest_edge() =
        video_preprocess_reader.value_or<int>("size.shortest_edge", 0);
    args->mm_video_longest_edge() =
        video_preprocess_reader.value_or<int>("size.longest_edge", 0);

    const nlohmann::json& video_preprocess_data =
        video_preprocess_reader.data();
    if (video_preprocess_reader.contains("image_mean")) {
      args->mm_video_normalize_mean() =
          video_preprocess_data["image_mean"].get<std::vector<double>>();
    }

    if (video_preprocess_reader.contains("image_std")) {
      args->mm_video_normalize_std() =
          video_preprocess_data["image_std"].get<std::vector<double>>();
    }
    args->mm_video_patch_size() =
        video_preprocess_reader.value_or<int>("patch_size", 0);
    args->mm_video_temporal_patch_size() =
        video_preprocess_reader.value_or<int>("temporal_patch_size", 0);
    args->mm_video_merge_size() =
        video_preprocess_reader.value_or<int>("merge_size", 0);
    args->mm_video_do_rescale() =
        video_preprocess_reader.value_or<bool>("do_rescale", false);
  }

  return true;
}

}  // namespace xllm
