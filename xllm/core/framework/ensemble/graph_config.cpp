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

std::unordered_map<std::string, std::string> parse_string_map(
    const nlohmann::json& root,
    const std::string& field_name) {
  std::unordered_map<std::string, std::string> output;
  if (!root.contains(field_name)) {
    return output;
  }
  const nlohmann::json& value = root.at(field_name);
  if (!value.is_object()) {
    LOG(FATAL) << field_name << " must be an object.";
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
    LOG(FATAL) << field_name << "." << it.key() << " must be a scalar value.";
  }
  return output;
}

std::vector<std::string> parse_string_array(const nlohmann::json& root,
                                            const std::string& field_name) {
  std::vector<std::string> output;
  if (!root.contains(field_name)) {
    return output;
  }
  const nlohmann::json& value = root.at(field_name);
  if (!value.is_array()) {
    LOG(FATAL) << field_name << " must be a string array.";
  }
  output.reserve(value.size());
  for (const nlohmann::json& item : value) {
    if (!item.is_string()) {
      LOG(FATAL) << field_name << " must be a string array.";
    }
    output.emplace_back(item.get<std::string>());
  }
  return output;
}

std::map<int32_t, int32_t> parse_ranks(const nlohmann::json& node_json) {
  std::map<int32_t, int32_t> ranks;
  if (!node_json.contains("ranks")) {
    return ranks;
  }
  if (!node_json.at("ranks").is_array()) {
    LOG(FATAL) << "node.ranks must be an integer array.";
  }
  const nlohmann::json& ranks_json = node_json.at("ranks");
  std::vector<int32_t> global_ranks;
  global_ranks.reserve(ranks_json.size());
  for (const nlohmann::json& item : ranks_json) {
    if (!item.is_number_integer()) {
      LOG(FATAL) << "node.ranks must be an integer array.";
    }
    int64_t rank = item.get<int64_t>();
    if (rank < 0 || rank > std::numeric_limits<int32_t>::max()) {
      LOG(FATAL) << "node.ranks contains invalid rank.";
    }
    global_ranks.emplace_back(static_cast<int32_t>(rank));
  }
  std::sort(global_ranks.begin(), global_ranks.end());
  for (std::size_t index = 0; index < global_ranks.size(); ++index) {
    const int32_t global_rank = global_ranks[index];
    if (index > 0 && global_rank == global_ranks[index - 1]) {
      LOG(FATAL) << "node.ranks contains duplicate rank.";
    }
    ranks.emplace(global_rank, static_cast<int32_t>(index));
  }
  return ranks;
}

EndpointConfig parse_endpoint_config(const nlohmann::json& node_json) {
  if (!node_json.contains("endpoint")) {
    LOG(FATAL) << "endpoint is required.";
  }
  const nlohmann::json& endpoint_json = node_json.at("endpoint");
  if (!endpoint_json.is_object()) {
    LOG(FATAL) << "endpoint must be an object.";
  }
  if (!endpoint_json.contains("transport") ||
      !endpoint_json.at("transport").is_string()) {
    LOG(FATAL) << "endpoint.transport must be a string.";
  }
  if (!endpoint_json.contains("target") ||
      !endpoint_json.at("target").is_string()) {
    LOG(FATAL) << "endpoint.target must be a string.";
  }

  EndpointConfig endpoint;
  endpoint.transport = endpoint_json.at("transport").get<std::string>();
  endpoint.target = endpoint_json.at("target").get<std::string>();
  endpoint.args = parse_string_map(endpoint_json, "args");
  return endpoint;
}

NodeConfig parse_node(const nlohmann::json& node_json) {
  if (!node_json.is_object()) {
    LOG(FATAL) << "nodes item must be an object.";
  }
  if (!node_json.contains("name") || !node_json.at("name").is_string()) {
    LOG(FATAL) << "node.name must be a string.";
  }
  if (!node_json.contains("node_type") ||
      !node_json.at("node_type").is_string()) {
    LOG(FATAL) << "node.node_type must be a string.";
  }

  NodeConfig node;
  node.name = node_json.at("name").get<std::string>();
  node.node_type = node_json.at("node_type").get<std::string>();
  node.deps = parse_string_array(node_json, "node.deps");
  node.ranks = parse_ranks(node_json);
  node.engine_config = parse_string_map(node_json, "engine_config");
  if (!node_json.contains("request_adapter") ||
      !node_json.at("request_adapter").is_string()) {
    LOG(FATAL) << "node.request_adapter must be a string.";
  }
  node.request_adapter = node_json.at("request_adapter").get<std::string>();
  node.request_adapter_config =
      parse_string_map(node_json, "request_adapter_config");
  if (!node_json.contains("result_adapter") ||
      !node_json.at("result_adapter").is_string()) {
    LOG(FATAL) << "node.result_adapter must be a string.";
  }
  node.result_adapter = node_json.at("result_adapter").get<std::string>();
  node.result_adapter_config =
      parse_string_map(node_json, "result_adapter_config");
  node.endpoint = parse_endpoint_config(node_json);
  if (node_json.contains("final_output")) {
    if (!node_json.at("final_output").is_boolean()) {
      LOG(FATAL) << "node.final_output must be a bool.";
    }
    node.final_output = node_json.at("final_output").get<bool>();
  }
  node.output_keys = parse_string_array(node_json, "node.output_keys");
  if (node_json.contains("timeout_ms")) {
    if (!node_json.at("timeout_ms").is_number_integer()) {
      LOG(FATAL) << "node.timeout_ms must be an integer.";
    }
    node.timeout_ms = node_json.at("timeout_ms").get<int64_t>();
    if (node.timeout_ms < 0) {
      LOG(FATAL) << "node.timeout_ms must be non-negative.";
    }
  }
  return node;
}

}  // namespace

void validate_graph_config(const GraphConfig& config) {
  std::unordered_set<std::string> node_names;
  std::unordered_set<int32_t> global_ranks;
  bool has_final_output = false;
  std::unordered_map<std::string, int32_t> indegrees;
  std::unordered_map<std::string, std::vector<std::string>> downstream_nodes;

  for (const NodeConfig& node : config.nodes) {
    if (node.name.empty()) {
      LOG(FATAL) << "node name cannot be empty.";
    }
    if (!node_names.insert(node.name).second) {
      LOG(FATAL) << "Duplicate node name: " << node.name;
    }
    if (node.ranks.empty()) {
      LOG(FATAL) << "node ranks cannot be empty: " << node.name;
    }
    int32_t expected_local_rank = 0;
    for (const auto& rank : node.ranks) {
      const int32_t global_rank = rank.first;
      const int32_t local_rank = rank.second;
      if (global_rank < 0) {
        LOG(FATAL) << "Global rank is out of range in node: " << node.name;
      }
      if (local_rank < 0 ||
          local_rank >= static_cast<int32_t>(node.ranks.size())) {
        LOG(FATAL) << "Local rank is out of range in node: " << node.name;
      }
      if (!global_ranks.insert(global_rank).second) {
        LOG(FATAL) << "Rank belongs to multiple nodes: " << global_rank;
      }
      if (local_rank != expected_local_rank) {
        LOG(FATAL) << "Local rank mapping is invalid in node: " << node.name;
      }
      ++expected_local_rank;
    }
    if (node.final_output) {
      has_final_output = true;
      if (node.output_keys.empty()) {
        LOG(FATAL) << "final output node output_keys cannot be empty: "
                   << node.name;
      }
    }
    indegrees.emplace(node.name, 0);
  }

  if (!has_final_output) {
    LOG(FATAL) << "At least one final_output node is required.";
  }
  for (const NodeConfig& node : config.nodes) {
    for (const std::string& dep : node.deps) {
      if (node_names.find(dep) == node_names.end()) {
        LOG(FATAL) << "deps references unknown node: " << dep;
      }
      downstream_nodes[dep].emplace_back(node.name);
      ++indegrees[node.name];
    }
    if (node.final_output &&
        downstream_nodes.find(node.name) != downstream_nodes.end()) {
      LOG(FATAL) << "final output node cannot have downstream node: "
                 << node.name;
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
    LOG(FATAL) << "graph must be a DAG.";
  }
}

void load_graph_config_from_file(const std::string& path, GraphConfig& config) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open graph config: " << path;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  load_graph_config_from_json(buffer.str(), config);
}

void load_graph_config_from_json(const std::string& json_text,
                                 GraphConfig& config) {
  GraphConfig parsed_config;
  try {
    nlohmann::json root = nlohmann::json::parse(json_text);
    if (!root.is_object()) {
      LOG(FATAL) << "graph config must be an object.";
    }
    if (!root.contains("graph_name") || !root.at("graph_name").is_string()) {
      LOG(FATAL) << "graph_name must be a string.";
    }
    parsed_config.graph_name = root.at("graph_name").get<std::string>();
    if (root.contains("ready_timeout_ms")) {
      if (!root.at("ready_timeout_ms").is_number_integer()) {
        LOG(FATAL) << "ready_timeout_ms must be an integer.";
      }
      parsed_config.ready_timeout_ms =
          root.at("ready_timeout_ms").get<int64_t>();
      if (parsed_config.ready_timeout_ms < 0) {
        LOG(FATAL) << "ready_timeout_ms must be non-negative.";
      }
    }
    if (!root.contains("nodes")) {
      LOG(FATAL) << "nodes is required.";
    }
    if (!root.at("nodes").is_array()) {
      LOG(FATAL) << "nodes must be an array.";
    }
    const nlohmann::json& nodes_json = root.at("nodes");
    parsed_config.nodes.reserve(nodes_json.size());
    for (const nlohmann::json& node_json : nodes_json) {
      parsed_config.nodes.emplace_back(parse_node(node_json));
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(FATAL) << "Failed to parse graph config: " << e.what();
  }
  config = std::move(parsed_config);
}

std::string make_ready_key(const std::string& graph_name,
                           const std::string& node_name) {
  return graph_name + "/" + node_name;
}

}  // namespace ensemble
}  // namespace xllm
