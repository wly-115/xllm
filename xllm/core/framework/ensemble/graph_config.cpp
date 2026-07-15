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

#include "core/framework/ensemble/graph_config.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xllm {
namespace {

constexpr char kVlmBackend[] = "vlm";
constexpr char kDitBackend[] = "dit";
constexpr char kQwenVlmEncodeAdapter[] = "qwen_vlm_encode";
constexpr char kQwenImageEditDitAdapter[] = "qwen_image_edit_dit";
constexpr char kQwenImageEditPreprocessor[] = "qwen-image-edit";

std::unordered_map<std::string, std::string> parse_string_map(
    const YAML::Node& root,
    const std::string& field_name) {
  std::unordered_map<std::string, std::string> output;
  const YAML::Node value = root[field_name];
  if (!value) {
    return output;
  }
  if (!value.IsMap()) {
    LOG(FATAL) << field_name << " must be a map.";
  }
  for (const auto& entry : value) {
    if (!entry.first.IsScalar() || !entry.second.IsScalar()) {
      LOG(FATAL) << field_name << " entries must be scalar values.";
    }
    output.emplace(entry.first.as<std::string>(), entry.second.Scalar());
  }
  return output;
}

std::vector<std::string> parse_string_array(const YAML::Node& root,
                                            const std::string& field_name) {
  std::vector<std::string> output;
  const YAML::Node value = root[field_name];
  if (!value) {
    return output;
  }
  if (!value.IsSequence()) {
    LOG(FATAL) << field_name << " must be a string sequence.";
  }
  output.reserve(value.size());
  for (const YAML::Node& item : value) {
    if (!item.IsScalar()) {
      LOG(FATAL) << field_name << " must be a string sequence.";
    }
    output.emplace_back(item.as<std::string>());
  }
  return output;
}

std::map<int32_t, int32_t> parse_ranks(const YAML::Node& node_yaml) {
  std::map<int32_t, int32_t> ranks;
  const YAML::Node ranks_yaml = node_yaml["ranks"];
  if (!ranks_yaml) {
    return ranks;
  }
  if (!ranks_yaml.IsSequence()) {
    LOG(FATAL) << "node.ranks must be an integer sequence.";
  }
  std::vector<int32_t> global_ranks;
  global_ranks.reserve(ranks_yaml.size());
  for (const YAML::Node& item : ranks_yaml) {
    const int64_t rank = item.as<int64_t>();
    if (rank < 0 || rank > std::numeric_limits<int32_t>::max()) {
      LOG(FATAL) << "node.ranks contains invalid rank.";
    }
    global_ranks.emplace_back(static_cast<int32_t>(rank));
  }
  std::sort(global_ranks.begin(), global_ranks.end());
  for (size_t index = 0; index < global_ranks.size(); ++index) {
    const int32_t global_rank = global_ranks[index];
    if (index > 0 && global_rank == global_ranks[index - 1]) {
      LOG(FATAL) << "node.ranks contains duplicate rank.";
    }
    ranks.emplace(global_rank, static_cast<int32_t>(index));
  }
  return ranks;
}

std::string parse_endpoint_target(const YAML::Node& node_yaml) {
  const YAML::Node endpoint_yaml = node_yaml["endpoint"];
  if (!endpoint_yaml) {
    LOG(FATAL) << "endpoint is required.";
  }
  if (!endpoint_yaml.IsMap()) {
    LOG(FATAL) << "endpoint must be a map.";
  }
  const YAML::Node target = endpoint_yaml["target"];
  if (!target || !target.IsScalar()) {
    LOG(FATAL) << "endpoint target must be a string.";
  }
  return target.as<std::string>();
}

std::string parse_result_target(const YAML::Node& root) {
  const YAML::Node endpoint_yaml = root["result_endpoint"];
  if (!endpoint_yaml) {
    LOG(FATAL) << "result_endpoint is required.";
  }
  if (!endpoint_yaml.IsMap()) {
    LOG(FATAL) << "result_endpoint must be a map.";
  }
  const YAML::Node target_yaml = endpoint_yaml["target"];
  if (!target_yaml || !target_yaml.IsScalar()) {
    LOG(FATAL) << "result_endpoint.target must be a string.";
  }
  const std::string target = target_yaml.as<std::string>();
  if (target.empty()) {
    LOG(FATAL) << "result_endpoint.target cannot be empty.";
  }
  return target;
}

NodeConfig parse_node(const YAML::Node& node_yaml) {
  if (!node_yaml.IsMap()) {
    LOG(FATAL) << "nodes item must be a map.";
  }
  if (!node_yaml["name"] || !node_yaml["name"].IsScalar()) {
    LOG(FATAL) << "node.name must be a string.";
  }

  NodeConfig node;
  node.name = node_yaml["name"].as<std::string>();
  if (!node_yaml["adapter"] || !node_yaml["adapter"].IsScalar()) {
    LOG(FATAL) << "node.adapter must be a string.";
  }
  node.adapter = node_yaml["adapter"].as<std::string>();
  node.deps = parse_string_array(node_yaml, "deps");
  node.ranks = parse_ranks(node_yaml);
  node.engine_config = parse_string_map(node_yaml, "engine_config");
  auto backend_it = node.engine_config.find("backend");
  if (backend_it == node.engine_config.end()) {
    LOG(FATAL) << "node.engine_config.backend is required: " << node.name;
  }
  node.backend = backend_it->second;
  node.endpoint_target = parse_endpoint_target(node_yaml);
  if (node_yaml["final_output"]) {
    node.final_output = node_yaml["final_output"].as<bool>();
  }
  if (node_yaml["timeout_ms"]) {
    node.timeout_ms = node_yaml["timeout_ms"].as<int64_t>();
    if (node.timeout_ms < 0) {
      LOG(FATAL) << "node.timeout_ms must be non-negative.";
    }
  }
  return node;
}

PreprocessorConfig parse_preprocessor_config(const YAML::Node& root) {
  const YAML::Node preprocessor_yaml = root["preprocessor_config"];
  if (!preprocessor_yaml.IsMap()) {
    LOG(FATAL) << "preprocessor_config must be a map.";
  }
  if (!preprocessor_yaml["type"] || !preprocessor_yaml["type"].IsScalar()) {
    LOG(FATAL) << "preprocessor_config.type must be a string.";
  }
  if (!preprocessor_yaml["model_root"] ||
      !preprocessor_yaml["model_root"].IsScalar()) {
    LOG(FATAL) << "preprocessor_config.model_root must be a string.";
  }

  PreprocessorConfig config;
  config.type = preprocessor_yaml["type"].as<std::string>();
  config.model_root = preprocessor_yaml["model_root"].as<std::string>();
  if (config.type.empty()) {
    LOG(FATAL) << "preprocessor_config.type cannot be empty.";
  }
  if (config.model_root.empty()) {
    LOG(FATAL) << "preprocessor_config.model_root cannot be empty.";
  }
  if (config.type != kQwenImageEditPreprocessor) {
    LOG(FATAL) << "Unsupported preprocessor_config.type: " << config.type;
  }
  return config;
}

struct GraphValidationContext {
  std::unordered_set<std::string> node_names;
  std::unordered_set<int32_t> global_ranks;
  std::unordered_map<std::string, std::vector<std::string>> downstream_nodes;
  std::vector<std::string> root_nodes;
  std::vector<std::string> final_nodes;
};

void validate_node_identity(const NodeConfig& node,
                            GraphValidationContext& context) {
  if (node.name.empty()) {
    LOG(FATAL) << "node name cannot be empty.";
  }
  if (!context.node_names.insert(node.name).second) {
    LOG(FATAL) << "Duplicate node name: " << node.name;
  }
}

void validate_node_ranks(const NodeConfig& node,
                         GraphValidationContext& context) {
  if (node.ranks.empty()) {
    LOG(FATAL) << "node ranks cannot be empty: " << node.name;
  }

  for (const auto& rank : node.ranks) {
    const int32_t global_rank = rank.first;
    if (!context.global_ranks.insert(global_rank).second) {
      LOG(FATAL) << "Rank belongs to multiple nodes: " << global_rank;
    }
  }
}

void validate_node_runtime_config(const NodeConfig& node) {
  if (node.backend.empty()) {
    LOG(FATAL) << "node backend cannot be empty: " << node.name;
  }
  if (node.adapter.empty()) {
    LOG(FATAL) << "node adapter cannot be empty: " << node.name;
  }
  if (node.endpoint_target.empty()) {
    LOG(FATAL) << "node endpoint target cannot be empty: " << node.name;
  }
  if (node.deps.size() > 1) {
    LOG(FATAL) << "P0 node supports at most one dependency: " << node.name;
  }

  const bool valid_vlm =
      node.backend == kVlmBackend && node.adapter == kQwenVlmEncodeAdapter;
  const bool valid_dit =
      node.backend == kDitBackend && node.adapter == kQwenImageEditDitAdapter;
  if (!valid_vlm && !valid_dit) {
    LOG(FATAL) << "Unsupported backend and adapter combination: backend="
               << node.backend << ", adapter=" << node.adapter;
  }
}

void validate_nodes(const GraphConfig& config,
                    GraphValidationContext& context) {
  for (const NodeConfig& node : config.nodes) {
    validate_node_identity(node, context);
    validate_node_ranks(node, context);
    validate_node_runtime_config(node);
    if (node.deps.empty()) {
      context.root_nodes.emplace_back(node.name);
    }
    if (node.final_output) {
      context.final_nodes.emplace_back(node.name);
    }
  }
}

void validate_dependencies(const GraphConfig& config,
                           GraphValidationContext& context) {
  for (const NodeConfig& node : config.nodes) {
    for (const std::string& dep : node.deps) {
      if (context.node_names.find(dep) == context.node_names.end()) {
        LOG(FATAL) << "deps references unknown node: " << dep;
      }
      context.downstream_nodes[dep].emplace_back(node.name);
    }
  }

  for (const auto& downstream_entry : context.downstream_nodes) {
    if (downstream_entry.second.size() > 1) {
      LOG(FATAL) << "P0 node supports at most one downstream node: "
                 << downstream_entry.first;
    }
  }
}

void validate_endpoints(const GraphConfig& config,
                        const GraphValidationContext& context) {
  if (config.result_target.empty()) {
    LOG(FATAL) << "result endpoint target cannot be empty.";
  }
  if (context.root_nodes.size() != 1) {
    LOG(FATAL) << "P0 graph must have exactly one root node.";
  }
  if (context.final_nodes.size() != 1) {
    LOG(FATAL) << "P0 graph must have exactly one final node.";
  }

  for (const NodeConfig& node : config.nodes) {
    if (node.final_output && context.downstream_nodes.find(node.name) !=
                                 context.downstream_nodes.end()) {
      LOG(FATAL) << "final output node cannot have downstream node: "
                 << node.name;
    }
  }
}

void validate_linear_chain(const GraphConfig& config,
                           const GraphValidationContext& context) {
  const std::string& final_node = context.final_nodes.front();
  std::string node_name = context.root_nodes.front();
  std::unordered_set<std::string> visited_nodes;
  while (visited_nodes.insert(node_name).second) {
    auto downstream_it = context.downstream_nodes.find(node_name);
    if (downstream_it == context.downstream_nodes.end()) {
      break;
    }
    node_name = downstream_it->second.front();
  }

  if (node_name != final_node) {
    LOG(FATAL) << "P0 graph chain must terminate at the unique final node.";
  }
  if (visited_nodes.size() != config.nodes.size()) {
    LOG(FATAL) << "P0 graph must be fully connected from root to final.";
  }
}

void validate_preprocessor_config(const GraphConfig& config) {
  if (!config.preprocessor_config.has_value()) {
    LOG(FATAL) << "qwen-image-edit graph requires preprocessor_config.";
  }

  const PreprocessorConfig& preprocessor_config = *config.preprocessor_config;
  if (preprocessor_config.type.empty()) {
    LOG(FATAL) << "preprocessor_config.type cannot be empty.";
  }
  if (preprocessor_config.model_root.empty()) {
    LOG(FATAL) << "preprocessor_config.model_root cannot be empty.";
  }
  if (preprocessor_config.type != kQwenImageEditPreprocessor) {
    LOG(FATAL) << "Unsupported preprocessor_config.type: "
               << preprocessor_config.type;
  }
}

}  // namespace

void GraphConfig::build_indices() {
  validate_graph_config(*this);

  node_index.clear();
  node_index.reserve(nodes.size());
  downstream_nodes.clear();
  downstream_nodes.resize(nodes.size());
  final_output_nodes.clear();
  final_output_nodes.reserve(nodes.size());
  root_nodes.clear();
  root_nodes.reserve(nodes.size());

  for (size_t index = 0; index < nodes.size(); ++index) {
    node_index.emplace(nodes[index].name, index);
  }

  for (const NodeConfig& node : nodes) {
    if (node.deps.empty()) {
      root_nodes.emplace_back(node.name);
    }
    if (node.final_output) {
      final_output_nodes.emplace_back(node.name);
    }
    for (const std::string& dep : node.deps) {
      auto dep_it = node_index.find(dep);
      downstream_nodes[dep_it->second].emplace_back(node.name);
    }
  }
}

const NodeConfig* GraphConfig::find_node(const std::string& name) const {
  auto it = node_index.find(name);
  if (it == node_index.end()) {
    return nullptr;
  }
  return &nodes[it->second];
}

const std::vector<std::string>& GraphConfig::get_downstream(
    const std::string& name) const {
  auto it = node_index.find(name);
  if (it == node_index.end()) {
    LOG(FATAL) << "Unknown node: " << name;
  }
  return downstream_nodes[it->second];
}

const std::vector<std::string>& GraphConfig::get_root_nodes() const {
  return root_nodes;
}

const std::vector<std::string>& GraphConfig::get_final_nodes() const {
  return final_output_nodes;
}

void validate_graph_config(const GraphConfig& config) {
  if (config.nodes.empty()) {
    LOG(FATAL) << "graph must contain at least one node.";
  }
  GraphValidationContext context;
  validate_nodes(config, context);
  validate_dependencies(config, context);
  validate_endpoints(config, context);
  validate_linear_chain(config, context);
  validate_preprocessor_config(config);
}

void load_graph_config_from_file(const std::string& path, GraphConfig& config) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open graph config: " << path;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  load_graph_config_from_yaml(buffer.str(), config);
}

void load_graph_config_from_yaml(const std::string& yaml_text,
                                 GraphConfig& config) {
  GraphConfig parsed_config;
  try {
    const YAML::Node root = YAML::Load(yaml_text);
    if (!root.IsMap()) {
      LOG(FATAL) << "graph config must be a map.";
    }
    if (!root["graph_name"] || !root["graph_name"].IsScalar()) {
      LOG(FATAL) << "graph_name must be a string.";
    }
    parsed_config.graph_name = root["graph_name"].as<std::string>();
    parsed_config.result_target = parse_result_target(root);
    if (root["preprocessor_config"]) {
      parsed_config.preprocessor_config = parse_preprocessor_config(root);
    }
    if (root["ready_timeout_ms"]) {
      parsed_config.ready_timeout_ms = root["ready_timeout_ms"].as<int64_t>();
      if (parsed_config.ready_timeout_ms < 0) {
        LOG(FATAL) << "ready_timeout_ms must be non-negative.";
      }
    }
    if (!root["nodes"]) {
      LOG(FATAL) << "nodes is required.";
    }
    const YAML::Node nodes_yaml = root["nodes"];
    if (!nodes_yaml.IsSequence()) {
      LOG(FATAL) << "nodes must be a sequence.";
    }
    parsed_config.nodes.reserve(nodes_yaml.size());
    for (const YAML::Node& node_yaml : nodes_yaml) {
      parsed_config.nodes.emplace_back(parse_node(node_yaml));
    }
  } catch (const YAML::Exception& e) {
    LOG(FATAL) << "Failed to parse graph config: " << e.what();
  }
  config = std::move(parsed_config);
  config.build_indices();
}

}  // namespace xllm
