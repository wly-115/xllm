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
#include <unordered_set>

#include "core/common/global_flags.h"
#include "core/distributed_runtime/ensemble_engine.h"
#include "core/distributed_runtime/ensemble_node_ready_service.h"
#include "core/framework/ensemble/graph.h"
#include "core/framework/ensemble/graph_config.h"
#include "server/xllm_server_registry.h"

namespace xllm {

OmniMaster::~OmniMaster() { stop_ready_service(); }

bool OmniMaster::start(const ensemble::GraphConfig& graph_config) {
  graph_ = std::make_shared<const ensemble::Graph>(
      ensemble::build_graph_from_config(graph_config));
  ensemble_engine_ = std::make_shared<EnsembleEngine>(graph_);
  ready_timeout_ms_ = graph_config.ready_timeout_ms;
  const int32_t total_num = static_cast<int32_t>(graph_->nodes().size());
  ready_service_ = std::make_shared<EnsembleNodeReadyService>(total_num);
  return start_service();
}

bool OmniMaster::wait() { return wait_ready(); }

bool OmniMaster::start_service() {
  if (FLAGS_omni_master_addr.empty()) {
    LOG(ERROR) << "omni_master_addr cannot be empty.";
    return false;
  }

  ready_server_name_ = "EnsembleNodeReady";
  auto& registry = ServerRegistry::get_instance();
  if (registry.try_get_server(ready_server_name_) != nullptr) {
    LOG(ERROR) << "EnsembleNodeReady server has already been registered.";
    return false;
  }

  XllmServer* ready_server = registry.register_server(ready_server_name_);
  if (!ready_server->start(
          ready_service_.get(), FLAGS_omni_master_addr, ready_server_name_)) {
    LOG(ERROR) << "Failed to start EnsembleNodeReady server on address "
               << FLAGS_omni_master_addr;
    registry.unregister_server(ready_server_name_);
    ready_server_name_.clear();
    return false;
  }

  LOG(INFO) << "Started EnsembleNodeReady server on address "
            << FLAGS_omni_master_addr;
  return true;
}

bool OmniMaster::wait_ready() {
  std::unordered_set<std::string> ready_nodes =
      ready_service_->wait(ready_timeout_ms_);
  const int32_t total_num = static_cast<int32_t>(graph_->nodes().size());
  if (static_cast<int32_t>(ready_nodes.size()) != total_num) {
    LOG(ERROR) << "Not all engine services are ready. expected=" << total_num
               << ", actual=" << ready_nodes.size();
    stop_ready_service();
    return false;
  }
  return true;
}

void OmniMaster::stop_ready_service() {
  if (ready_server_name_.empty()) {
    return;
  }

  ServerRegistry::get_instance().unregister_server(ready_server_name_);
  ready_server_name_.clear();
  ready_service_.reset();
}

}  // namespace xllm
