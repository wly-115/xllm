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
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace ensemble {

struct EndpointConfig {
  std::string transport;
  std::string target;
  std::unordered_map<std::string, std::string> args;
};

struct NodeConfig {
  std::string name;
  std::string node_type;
  std::vector<std::string> deps;
  std::map<int32_t, int32_t> ranks;
  std::unordered_map<std::string, std::string> engine_config;
  std::string request_adapter;
  std::unordered_map<std::string, std::string> request_adapter_config;
  std::string result_adapter;
  std::unordered_map<std::string, std::string> result_adapter_config;
  EndpointConfig endpoint;
  bool final_output = false;
  std::vector<std::string> output_keys;
  int64_t timeout_ms = 30000;
};

struct GraphConfig {
  std::string graph_name;
  int64_t ready_timeout_ms = 60000;
  std::vector<NodeConfig> nodes;
};

void load_graph_config_from_file(const std::string& path, GraphConfig& config);

void load_graph_config_from_json(const std::string& json_text,
                                 GraphConfig& config);

void validate_graph_config(const GraphConfig& config);

std::string make_ready_key(const std::string& graph_name,
                           const std::string& node_name);

}  // namespace ensemble
}  // namespace xllm
