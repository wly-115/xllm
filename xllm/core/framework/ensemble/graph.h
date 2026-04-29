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
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/ensemble/graph_config.h"

namespace xllm {
namespace ensemble {

class Node final {
 public:
  explicit Node(NodeConfig config,
                int32_t leader_rank,
                std::string ready_key,
                std::vector<std::string> downstream_nodes);

  const NodeConfig& config() const;
  const std::string& name() const;
  int32_t leader_rank() const;
  const std::string& ready_key() const;
  const std::vector<std::string>& deps() const;
  const std::vector<std::string>& downstream_nodes() const;
  bool final_output() const;
  const std::vector<std::string>& output_keys() const;

 private:
  NodeConfig config_;
  int32_t leader_rank_ = -1;
  std::string ready_key_;
  std::vector<std::string> downstream_nodes_;
};

class Graph final {
 public:
  explicit Graph(std::string graph_name,
                 std::vector<Node> nodes,
                 std::unordered_map<std::string, std::size_t> node_index,
                 std::vector<std::string> final_output_nodes);

  const std::string& graph_name() const;
  const std::vector<Node>& nodes() const;
  std::optional<std::reference_wrapper<const Node>> find_node(
      const std::string& node_name) const;
  std::optional<std::reference_wrapper<const std::vector<std::string>>> deps(
      const std::string& node_name) const;
  std::optional<std::reference_wrapper<const std::vector<std::string>>>
  downstream_nodes(const std::string& node_name) const;
  const std::vector<std::string>& final_output_nodes() const;
  std::optional<std::reference_wrapper<const std::vector<std::string>>>
  final_output_keys(const std::string& node_name) const;

 private:
  std::string graph_name_;
  std::vector<Node> nodes_;
  std::unordered_map<std::string, std::size_t> node_index_;
  std::vector<std::string> final_output_nodes_;
};

Graph build_graph_from_config(const GraphConfig& config);

}  // namespace ensemble
}  // namespace xllm
