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

#include <algorithm>
#include <cstddef>
#include <deque>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xllm {
namespace ensemble {
namespace {

Status invalid_argument(const std::string& message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

Status parse_string_map(const nlohmann::json& root,
                        const std::string& field_name,
                        std::unordered_map<std::string, std::string>& output) {
  output.clear();
  if (!root.contains(field_name)) {
    return Status();
  }
  const nlohmann::json& value = root.at(field_name);
  if (!value.is_object()) {
    return invalid_argument(field_name + " must be an object.");
  }
  for (auto it = value.begin(); it != value.end(); ++it) {
    const nlohmann::json& item = it.value();
    if (item.is_string()) {
      output.emplace(it.key(), item.get<std::string>());
      continue;
    }
    if (item.is_boolean() || item.is_number()) {
      output.emplace(it.key(), item.dump());
      continue;
    }
    return invalid_argument(field_name + "." + it.key() +
                            " must be a scalar value.");
  }
  return Status();
}

Status parse_node(const nlohmann::json& node_json, NodeConfig& node) {
  if (!node_json.is_object()) {
    return invalid_argument("nodes item must be an object.");
  }
  if (!node_json.contains("name") || !node_json.at("name").is_string()) {
    return invalid_argument("node.name must be a string.");
  }
  if (!node_json.contains("node_type") ||
      !node_json.at("node_type").is_string()) {
    return invalid_argument("node.node_type must be a string.");
  }
  node.name = node_json.at("name").get<std::string>();
  node.node_type = node_json.at("node_type").get<std::string>();
  if (node_json.contains("deps")) {
    if (!node_json.at("deps").is_array()) {
      return invalid_argument("node.deps must be a string array.");
    }
    for (const nlohmann::json& dep : node_json.at("deps")) {
      if (!dep.is_string()) {
        return invalid_argument("node.deps must be a string array.");
      }
      node.deps.emplace_back(dep.get<std::string>());
    }
  }
  node.ranks.clear();
  if (node_json.contains("ranks")) {
    if (!node_json.at("ranks").is_array()) {
      return invalid_argument("node.ranks must be an integer array.");
    }
    const nlohmann::json& ranks_json = node_json.at("ranks");
    std::vector<int32_t> global_ranks;
    global_ranks.reserve(ranks_json.size());
    for (const nlohmann::json& item : ranks_json) {
      if (!item.is_number_integer()) {
        return invalid_argument("node.ranks must be an integer array.");
      }
      int64_t rank = item.get<int64_t>();
      if (rank < 0 || rank > std::numeric_limits<int32_t>::max()) {
        return invalid_argument("node.ranks contains invalid rank.");
      }
      global_ranks.emplace_back(static_cast<int32_t>(rank));
    }
    std::sort(global_ranks.begin(), global_ranks.end());
    node.ranks.reserve(global_ranks.size());
    for (std::size_t index = 0; index < global_ranks.size(); ++index) {
      const int32_t global_rank = global_ranks[index];
      if (index > 0 && global_rank == global_ranks[index - 1]) {
        return invalid_argument("node.ranks contains duplicate rank.");
      }
      node.ranks.emplace(global_rank, static_cast<int32_t>(index));
    }
  }
  Status status =
      parse_string_map(node_json, "engine_config", node.engine_config);
  if (!status.ok()) {
    return status;
  }
  if (!node_json.contains("request_adapter") ||
      !node_json.at("request_adapter").is_string()) {
    return invalid_argument("node.request_adapter must be a string.");
  }
  node.request_adapter = node_json.at("request_adapter").get<std::string>();
  status = parse_string_map(
      node_json, "request_adapter_config", node.request_adapter_config);
  if (!status.ok()) {
    return status;
  }
  if (!node_json.contains("result_adapter") ||
      !node_json.at("result_adapter").is_string()) {
    return invalid_argument("node.result_adapter must be a string.");
  }
  node.result_adapter = node_json.at("result_adapter").get<std::string>();
  status = parse_string_map(
      node_json, "result_adapter_config", node.result_adapter_config);
  if (!status.ok()) {
    return status;
  }
  if (!node_json.contains("endpoint")) {
    return invalid_argument("endpoint is required.");
  }
  const nlohmann::json& endpoint_json = node_json.at("endpoint");
  if (!endpoint_json.is_object()) {
    return invalid_argument("endpoint must be an object.");
  }
  if (!endpoint_json.contains("transport") ||
      !endpoint_json.at("transport").is_string()) {
    return invalid_argument("endpoint.transport must be a string.");
  }
  if (!endpoint_json.contains("target") ||
      !endpoint_json.at("target").is_string()) {
    return invalid_argument("endpoint.target must be a string.");
  }
  node.endpoint.transport = endpoint_json.at("transport").get<std::string>();
  node.endpoint.target = endpoint_json.at("target").get<std::string>();
  status = parse_string_map(endpoint_json, "args", node.endpoint.args);
  if (!status.ok()) {
    return status;
  }
  if (node_json.contains("final_output")) {
    if (!node_json.at("final_output").is_boolean()) {
      return invalid_argument("node.final_output must be a bool.");
    }
    node.final_output = node_json.at("final_output").get<bool>();
  }
  if (node_json.contains("output_keys")) {
    if (!node_json.at("output_keys").is_array()) {
      return invalid_argument("node.output_keys must be a string array.");
    }
    for (const nlohmann::json& output_key : node_json.at("output_keys")) {
      if (!output_key.is_string()) {
        return invalid_argument("node.output_keys must be a string array.");
      }
      node.output_keys.emplace_back(output_key.get<std::string>());
    }
  }
  if (node_json.contains("timeout_ms")) {
    if (!node_json.at("timeout_ms").is_number_integer()) {
      return invalid_argument("node.timeout_ms must be an integer.");
    }
    node.timeout_ms = node_json.at("timeout_ms").get<int64_t>();
    if (node.timeout_ms < 0) {
      return invalid_argument("node.timeout_ms must be non-negative.");
    }
  }
  return Status();
}

}  // namespace

Status validate_graph_config(const GraphConfig& config) {
  std::unordered_set<std::string> node_names;
  std::unordered_set<int32_t> global_ranks;
  bool has_final_output = false;
  std::unordered_set<std::string> upstream_nodes;
  std::unordered_map<std::string, int32_t> indegrees;
  std::unordered_map<std::string, std::vector<std::string>> downstream_nodes;

  for (const NodeConfig& node : config.nodes) {
    if (node.name.empty()) {
      return invalid_argument("node name cannot be empty.");
    }
    if (!node_names.insert(node.name).second) {
      return invalid_argument("Duplicate node name: " + node.name);
    }
    std::unordered_set<int32_t> local_ranks;
    for (const auto& rank : node.ranks) {
      const int32_t global_rank = rank.first;
      const int32_t local_rank = rank.second;
      if (global_rank < 0) {
        return invalid_argument("Global rank is out of range in node: " +
                                node.name);
      }
      if (local_rank < 0 ||
          local_rank >= static_cast<int32_t>(node.ranks.size())) {
        return invalid_argument("Local rank is out of range in node: " +
                                node.name);
      }
      if (!local_ranks.insert(local_rank).second) {
        return invalid_argument("Duplicate local rank in node: " + node.name);
      }
      if (!global_ranks.insert(global_rank).second) {
        return invalid_argument("Rank belongs to multiple nodes: " +
                                std::to_string(global_rank));
      }
    }
    if (node.final_output) {
      has_final_output = true;
      if (node.output_keys.empty()) {
        return invalid_argument(
            "final output node output_keys cannot be "
            "empty: " +
            node.name);
      }
    }
    for (const std::string& dep : node.deps) {
      upstream_nodes.insert(dep);
    }
    indegrees.emplace(node.name, 0);
  }

  if (!has_final_output) {
    return invalid_argument("At least one final_output node is required.");
  }
  for (const NodeConfig& node : config.nodes) {
    for (const std::string& dep : node.deps) {
      if (node_names.find(dep) == node_names.end()) {
        return invalid_argument("deps references unknown node: " + dep);
      }
      downstream_nodes[dep].emplace_back(node.name);
      ++indegrees[node.name];
    }
    if (node.final_output &&
        upstream_nodes.find(node.name) != upstream_nodes.end()) {
      return invalid_argument(
          "final output node cannot have downstream node: " + node.name);
    }
  }
  std::deque<std::string> ready_nodes;
  for (const auto& [node_name, indegree] : indegrees) {
    if (indegree == 0) {
      ready_nodes.emplace_back(node_name);
    }
  }
  int32_t visited_count = 0;
  while (!ready_nodes.empty()) {
    std::string node_name = ready_nodes.front();
    ready_nodes.pop_front();
    ++visited_count;
    auto downstream_it = downstream_nodes.find(node_name);
    if (downstream_it == downstream_nodes.end()) {
      continue;
    }
    for (const std::string& downstream_node : downstream_it->second) {
      int32_t& indegree = indegrees[downstream_node];
      --indegree;
      if (indegree == 0) {
        ready_nodes.emplace_back(downstream_node);
      }
    }
  }
  if (visited_count != static_cast<int32_t>(config.nodes.size())) {
    return invalid_argument("graph must be a DAG.");
  }
  return Status();
}

Status load_graph_config_from_file(const std::string& path,
                                   GraphConfig& config) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return invalid_argument("Failed to open graph config: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return load_graph_config_from_json(buffer.str(), config);
}

Status load_graph_config_from_json(const std::string& json_text,
                                   GraphConfig& config) {
  GraphConfig parsed_config;
  try {
    nlohmann::json root = nlohmann::json::parse(json_text);
    if (!root.is_object()) {
      return invalid_argument("graph config must be an object.");
    }
    if (!root.contains("graph_name") || !root.at("graph_name").is_string()) {
      return invalid_argument("graph_name must be a string.");
    }
    parsed_config.graph_name = root.at("graph_name").get<std::string>();
    if (root.contains("ready_timeout_ms")) {
      if (!root.at("ready_timeout_ms").is_number_integer()) {
        return invalid_argument("ready_timeout_ms must be an integer.");
      }
      parsed_config.ready_timeout_ms =
          root.at("ready_timeout_ms").get<int64_t>();
      if (parsed_config.ready_timeout_ms < 0) {
        return invalid_argument("ready_timeout_ms must be non-negative.");
      }
    }
    if (!root.contains("nodes")) {
      return invalid_argument("nodes is required.");
    }
    if (!root.at("nodes").is_array()) {
      return invalid_argument("nodes must be an array.");
    }
    const nlohmann::json& nodes_json = root.at("nodes");
    parsed_config.nodes.reserve(nodes_json.size());
    for (const nlohmann::json& node_json : nodes_json) {
      NodeConfig node;
      Status status = parse_node(node_json, node);
      if (!status.ok()) {
        return status;
      }
      parsed_config.nodes.emplace_back(std::move(node));
    }
  } catch (const nlohmann::json::exception& e) {
    return invalid_argument(std::string("Failed to parse graph config: ") +
                            e.what());
  }
  config = std::move(parsed_config);
  return Status();
}

std::string make_ready_key(const std::string& graph_name,
                           const std::string& node_name) {
  return graph_name + "/" + node_name;
}

}  // namespace ensemble
}  // namespace xllm
