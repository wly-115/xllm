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

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace xllm {

struct NodeConfig {
  std::string name;
  std::string backend;
  std::string adapter;
  std::vector<std::string> deps;
  std::map<int32_t, int32_t> ranks;
  std::unordered_map<std::string, std::string> engine_config;
  std::string endpoint_target;
  bool final_output = false;
  int64_t timeout_ms = 30000;
};

struct PreprocessorConfig {
  std::string type;
  std::string model_root;
};

class GraphConfig final {
 public:
  std::string graph_name;
  std::string result_target;
  int64_t ready_timeout_ms = 60000;
  std::optional<PreprocessorConfig> preprocessor_config;
  std::vector<NodeConfig> nodes;
  absl::flat_hash_map<std::string, size_t> node_index;
  std::vector<std::vector<std::string>> downstream_nodes;
  std::vector<std::string> final_output_nodes;
  std::vector<std::string> root_nodes;

  void build_indices();
  const NodeConfig* find_node(const std::string& name) const;
  const std::vector<std::string>& get_downstream(const std::string& name) const;
  const std::vector<std::string>& get_root_nodes() const;
  const std::vector<std::string>& get_final_nodes() const;
};

void load_graph_config_from_file(const std::string& path, GraphConfig& config);

void load_graph_config_from_yaml(const std::string& yaml_text,
                                 GraphConfig& config);

void validate_graph_config(const GraphConfig& config);

}  // namespace xllm
