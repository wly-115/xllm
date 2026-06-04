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

#include "core/framework/ensemble/graph.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {
namespace ensemble {

Node::Node(NodeConfig config,
           int32_t leader_rank,
           std::string ready_key,
           std::vector<std::string> downstream_nodes)
    : config_(std::move(config)),
      leader_rank_(leader_rank),
      ready_key_(std::move(ready_key)),
      downstream_nodes_(std::move(downstream_nodes)) {}

const NodeConfig& Node::config() const { return config_; }

const std::string& Node::name() const { return config_.name; }

int32_t Node::leader_rank() const { return leader_rank_; }

const std::string& Node::ready_key() const { return ready_key_; }

const std::vector<std::string>& Node::deps() const { return config_.deps; }

const std::vector<std::string>& Node::downstream_nodes() const {
  return downstream_nodes_;
}

bool Node::final_output() const { return config_.final_output; }

const std::vector<std::string>& Node::output_keys() const {
  return config_.output_keys;
}

Graph::Graph(std::string graph_name,
             std::vector<Node> nodes,
             std::unordered_map<std::string, std::size_t> node_index,
             std::vector<std::string> final_output_nodes)
    : graph_name_(std::move(graph_name)),
      nodes_(std::move(nodes)),
      node_index_(std::move(node_index)),
      final_output_nodes_(std::move(final_output_nodes)) {}

const std::string& Graph::graph_name() const { return graph_name_; }

const std::vector<Node>& Graph::nodes() const { return nodes_; }

std::optional<std::reference_wrapper<const Node>> Graph::find_node(
    const std::string& node_name) const {
  auto it = node_index_.find(node_name);
  if (it == node_index_.end()) {
    return std::nullopt;
  }
  return std::cref(nodes_[it->second]);
}

std::optional<std::reference_wrapper<const std::vector<std::string>>>
Graph::deps(const std::string& node_name) const {
  std::optional<std::reference_wrapper<const Node>> node = find_node(node_name);
  if (!node.has_value()) {
    return std::nullopt;
  }
  return std::cref(node.value().get().deps());
}

std::optional<std::reference_wrapper<const std::vector<std::string>>>
Graph::downstream_nodes(const std::string& node_name) const {
  std::optional<std::reference_wrapper<const Node>> node = find_node(node_name);
  if (!node.has_value()) {
    return std::nullopt;
  }
  return std::cref(node.value().get().downstream_nodes());
}

const std::vector<std::string>& Graph::final_output_nodes() const {
  return final_output_nodes_;
}

std::optional<std::reference_wrapper<const std::vector<std::string>>>
Graph::final_output_keys(const std::string& node_name) const {
  std::optional<std::reference_wrapper<const Node>> node = find_node(node_name);
  if (!node.has_value() || !node.value().get().final_output()) {
    return std::nullopt;
  }
  return std::cref(node.value().get().output_keys());
}

Graph build_graph_from_config(const GraphConfig& config) {
  validate_graph_config(config);

  std::unordered_map<std::string, std::size_t> node_index;
  node_index.reserve(config.nodes.size());
  for (std::size_t index = 0; index < config.nodes.size(); ++index) {
    node_index.emplace(config.nodes[index].name, index);
  }

  std::vector<std::vector<std::string>> downstream_nodes;
  downstream_nodes.resize(config.nodes.size());
  for (const NodeConfig& node : config.nodes) {
    for (const std::string& dep : node.deps) {
      auto dep_it = node_index.find(dep);
      downstream_nodes[dep_it->second].emplace_back(node.name);
    }
  }

  std::vector<Node> nodes;
  nodes.reserve(config.nodes.size());
  std::vector<std::string> final_output_nodes;
  final_output_nodes.reserve(config.nodes.size());
  for (std::size_t index = 0; index < config.nodes.size(); ++index) {
    const NodeConfig& node_config = config.nodes[index];
    if (node_config.final_output) {
      final_output_nodes.emplace_back(node_config.name);
    }
    nodes.emplace_back(NodeConfig(node_config),
                       node_config.ranks.begin()->first,
                       make_ready_key(config.graph_name, node_config.name),
                       std::move(downstream_nodes[index]));
  }

  return Graph(config.graph_name,
               std::move(nodes),
               std::move(node_index),
               std::move(final_output_nodes));
}

}  // namespace ensemble
}  // namespace xllm
