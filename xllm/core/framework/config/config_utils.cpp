/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/config_utils.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"

DEFINE_string(config_json_file,
              "",
              "Path to a JSON config file. Values in the file override "
              "command-line flag values.");

DEFINE_bool(enable_dump_config_json,
            false,
            "Whether to dump the resolved startup config as JSON.");

DEFINE_string(dump_config_json_file,
              "xllm_config.json",
              "Path to write the resolved startup config as JSON. Used only "
              "when enable_dump_config_json is true.");

namespace xllm::config {
namespace {

std::mutex& parsed_json_config_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<std::once_flag>& parsed_json_config_once() {
  static std::unique_ptr<std::once_flag> once_flag =
      std::make_unique<std::once_flag>();
  return once_flag;
}

std::string& parsed_json_config_path() {
  static std::string config_path;
  return config_path;
}

std::optional<JsonReader>& parsed_json_config() {
  static std::optional<JsonReader> json_config;
  return json_config;
}

std::unordered_map<std::string, std::string>& runtime_config_overrides() {
  static std::unordered_map<std::string, std::string> overrides;
  return overrides;
}

nlohmann::json flag_value_to_json(
    const google::CommandLineFlagInfo& flag_info) {
  if (flag_info.type == "string") {
    return flag_info.current_value;
  }
  if (flag_info.type == "bool") {
    return flag_info.current_value == "true";
  }
  return nlohmann::json::parse(flag_info.current_value);
}

void apply_runtime_config_flags() {
  for (const auto& [name, value] : runtime_config_overrides()) {
    google::CommandLineFlagInfo flag_info;
    if (!google::GetCommandLineFlagInfo(name.c_str(), &flag_info)) {
      LOG(WARNING) << "Ignore unknown runtime config override: " << name;
      continue;
    }
    google::SetCommandLineOption(name.c_str(), value.c_str());
  }
}

void apply_runtime_config_overrides(nlohmann::json* config_json) {
  CHECK(config_json != nullptr) << "config json cannot be null.";
  for (const auto& [name, value] : runtime_config_overrides()) {
    google::CommandLineFlagInfo flag_info;
    if (!google::GetCommandLineFlagInfo(name.c_str(), &flag_info)) {
      LOG(WARNING) << "Ignore unknown runtime config override: " << name;
      continue;
    }

    google::GetCommandLineFlagInfo(name.c_str(), &flag_info);
    try {
      (*config_json)[name] = flag_value_to_json(flag_info);
    } catch (const nlohmann::json::exception& exception) {
      LOG(FATAL) << "Failed to parse runtime config override " << name << "="
                 << value << ", error: " << exception.what();
    }
  }
}

void load_parsed_json_config() {
  const std::string& config_path = parsed_json_config_path();
  nlohmann::json config_json = nlohmann::json::object();
  bool has_config = false;
  if (!config_path.empty()) {
    JsonReader reader;
    try {
      if (!reader.parse(config_path)) {
        LOG(ERROR) << "Failed to load JSON config file: " << config_path;
      } else {
        config_json = reader.data();
        has_config = true;
      }
    } catch (const std::exception& exception) {
      LOG(ERROR) << "Failed to parse JSON config file: " << config_path
                 << ", error: " << exception.what();
    }
  }

  apply_runtime_config_overrides(&config_json);
  if (!has_config && runtime_config_overrides().empty()) {
    return;
  }

  JsonReader merged_reader;
  merged_reader.parse_text(config_json.dump());
  parsed_json_config() = std::move(merged_reader);
}

void reset_parsed_json_config_if_path_changed() {
  if (parsed_json_config_path() == FLAGS_config_json_file) {
    return;
  }

  parsed_json_config_path() = FLAGS_config_json_file;
  parsed_json_config().reset();
  parsed_json_config_once() = std::make_unique<std::once_flag>();
}

nlohmann::ordered_json build_startup_config_json() {
  nlohmann::ordered_json config_json = nlohmann::ordered_json::object();

  ServiceConfig::get_instance().append_config_json(config_json);
  ModelConfig::get_instance().append_config_json(config_json);
  LoadConfig::get_instance().append_config_json(config_json);
  KVCacheConfig::get_instance().append_config_json(config_json);
  KVCacheStoreConfig::get_instance().append_config_json(config_json);
  BeamSearchConfig::get_instance().append_config_json(config_json);
  SchedulerConfig::get_instance().append_config_json(config_json);
  ParallelConfig::get_instance().append_config_json(config_json);
  EPLBConfig::get_instance().append_config_json(config_json);
  DistributedConfig::get_instance().append_config_json(config_json);
  DisaggPDConfig::get_instance().append_config_json(config_json);
  SpeculativeConfig::get_instance().append_config_json(config_json);
  ProfileConfig::get_instance().append_config_json(config_json);
  ExecutionConfig::get_instance().append_config_json(config_json);
  KernelConfig::get_instance().append_config_json(config_json);
  DiTConfig::get_instance().append_config_json(config_json);
  RecConfig::get_instance().append_config_json(config_json);

  return config_json;
}

}  // namespace

JsonReader load_json_file(const std::string& config_path) {
  JsonReader reader;
  if (!config_path.empty()) {
    reader.parse(config_path);
  }
  return reader;
}

JsonReader parse_json_string(std::string_view config_json) {
  JsonReader reader;
  if (!config_json.empty()) {
    reader.parse_text(std::string(config_json));
  }
  return reader;
}

bool is_flag_specified(const char* flag_name) {
  google::CommandLineFlagInfo flag_info;
  if (!google::GetCommandLineFlagInfo(flag_name, &flag_info)) {
    return false;
  }
  return !flag_info.is_default;
}

const std::optional<JsonReader>& get_parsed_json_config() {
  std::lock_guard<std::mutex> lock(parsed_json_config_mutex());
  reset_parsed_json_config_if_path_changed();
  std::call_once(*parsed_json_config_once(), load_parsed_json_config);
  return parsed_json_config();
}

void set_runtime_config_overrides(
    const std::unordered_map<std::string, std::string>& overrides) {
  std::lock_guard<std::mutex> lock(parsed_json_config_mutex());
  runtime_config_overrides() = overrides;
  apply_runtime_config_flags();
  parsed_json_config().reset();
  parsed_json_config_once() = std::make_unique<std::once_flag>();
}

void dump_startup_config() {
  if (!FLAGS_enable_dump_config_json) {
    return;
  }

  const std::filesystem::path dump_path =
      std::filesystem::path(FLAGS_dump_config_json_file).lexically_normal();
  if (dump_path.has_parent_path()) {
    std::error_code error_code;
    std::filesystem::create_directories(dump_path.parent_path(), error_code);
    if (error_code) {
      LOG(FATAL) << "Failed to create startup config dump directory: "
                 << dump_path.parent_path().string()
                 << ", error: " << error_code.message();
    }
  }

  std::ofstream output_stream(dump_path);
  if (!output_stream.is_open()) {
    LOG(FATAL) << "Failed to open startup config dump file: "
               << dump_path.string();
  }

  const nlohmann::ordered_json config_json = build_startup_config_json();
  output_stream << config_json.dump(2) << "\n";
  output_stream.close();
  if (!output_stream.good()) {
    LOG(FATAL) << "Failed to write startup config dump file: "
               << dump_path.string();
  }

  LOG(INFO) << "Dumped startup config to " << dump_path.string();
}

}  // namespace xllm::config
