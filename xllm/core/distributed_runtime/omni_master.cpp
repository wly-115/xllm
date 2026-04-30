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

#include "core/distributed_runtime/omni_master.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/distributed_runtime/ensemble_engine.h"
#include "core/distributed_runtime/ensemble_node_ready_service.h"
#include "core/framework/ensemble/graph.h"
#include "core/framework/ensemble/graph_config.h"

namespace xllm {

bool OmniMaster::init(const std::string& graph_config_path, int32_t node_rank) {
  if (!validate_init_args(graph_config_path, node_rank)) {
    return false;
  }
  if (!load_graph_config(graph_config_path)) {
    return false;
  }
  if (!wait_engine_services_ready()) {
    return false;
  }

  return create_ensemble_engine(build_graph());
}

bool OmniMaster::validate_init_args(const std::string& graph_config_path,
                                    int32_t node_rank) const {
  if (graph_config_path.empty()) {
    LOG(ERROR) << "graph_config_path cannot be empty.";
    return false;
  }
  if (node_rank < 0) {
    LOG(ERROR) << "node_rank must be non-negative.";
    return false;
  }
  return true;
}

bool OmniMaster::load_graph_config(const std::string& graph_config_path) {
  ensemble::GraphConfig graph_config;
  if (!ensemble::load_graph_config_from_file(graph_config_path, graph_config)) {
    return false;
  }

  if (!ensemble::validate_graph_config(graph_config)) {
    return false;
  }

  graph_config_ = std::move(graph_config);
  return true;
}

bool OmniMaster::wait_engine_services_ready() {
  const int32_t total_num = static_cast<int32_t>(graph_config_.nodes.size());
  ready_service_ = std::make_shared<EnsembleNodeReadyService>(total_num);
  std::unordered_map<std::string, std::string> ready_targets =
      ready_service_->wait(graph_config_.ready_timeout_ms);
  if (static_cast<int32_t>(ready_targets.size()) != total_num) {
    LOG(ERROR) << "Not all engine services are ready. expected=" << total_num
               << ", actual=" << ready_targets.size();
    return false;
  }

  for (ensemble::NodeConfig& node : graph_config_.nodes) {
    auto ready_it = ready_targets.find(node.name);
    if (ready_it == ready_targets.end()) {
      LOG(ERROR) << "Missing ready target for node: " << node.name;
      return false;
    }
    node.endpoint.target = ready_it->second;
  }
  return true;
}

std::shared_ptr<const ensemble::Graph> OmniMaster::build_graph() const {
  return std::make_shared<const ensemble::Graph>(
      ensemble::build_graph_from_config(graph_config_));
}

bool OmniMaster::create_ensemble_engine(
    std::shared_ptr<const ensemble::Graph> graph) {
  if (graph == nullptr) {
    LOG(ERROR) << "graph cannot be null.";
    return false;
  }

  ensemble_engine_ = std::make_shared<EnsembleEngine>(std::move(graph));
  return true;
}

}  // namespace xllm
