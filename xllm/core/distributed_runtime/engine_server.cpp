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

#include "core/distributed_runtime/engine_server.h"

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <utility>

#include "core/common/global_flags.h"
#include "core/distributed_runtime/dit_engine.h"
#include "core/distributed_runtime/vlm_engine.h"
#include "core/framework/ensemble/engine_config.h"
#include "server/xllm_server_registry.h"
#include "worker.pb.h"

namespace xllm {
namespace {

constexpr int32_t kReadyRegisterTimeoutMs = 10000;
constexpr int32_t kReadyRegisterMaxRetry = 3;

bool find_node_config(const ensemble::GraphConfig& graph_config,
                      int32_t node_rank,
                      ensemble::NodeConfig& node_config) {
  for (const ensemble::NodeConfig& candidate : graph_config.nodes) {
    if (candidate.ranks.find(node_rank) != candidate.ranks.end()) {
      node_config = candidate;
      return true;
    }
  }
  LOG(ERROR) << "node_rank does not belong to any graph node: " << node_rank;
  return false;
}

}  // namespace

EngineServer::~EngineServer() {
  if (server_name_.empty()) {
    return;
  }
  ServerRegistry::get_instance().unregister_server(server_name_);
}

bool EngineServer::init(const std::string& graph_config_path,
                        int32_t node_rank) {
  ensemble::GraphConfig graph_config;
  if (!load_graph_config(graph_config_path, graph_config)) {
    return false;
  }
  return init(graph_config, node_rank);
}

bool EngineServer::init(const ensemble::GraphConfig& graph_config,
                        int32_t node_rank) {
  if (node_rank < 0) {
    LOG(ERROR) << "node_rank must be non-negative.";
    return false;
  }
  if (!ensemble::validate_graph_config(graph_config)) {
    return false;
  }

  if (!select_node(graph_config, node_rank)) {
    return false;
  }
  if (!ensemble::initialize_engine_options_from_config(
          graph_config, node_rank, options_)) {
    return false;
  }
  if (!create_engine()) {
    return false;
  }

  if (!engine_->init()) {
    LOG(ERROR) << "Failed to initialize engine for graph node "
               << node_config_.name << ", backend=" << options_.backend();
    return false;
  }
  if (!init_engine_service(node_rank)) {
    return false;
  }

  return true;
}

EngineService* EngineServer::service() const { return service_.get(); }

Engine* EngineServer::engine() const { return engine_.get(); }

bool EngineServer::load_graph_config(
    const std::string& graph_config_path,
    ensemble::GraphConfig& graph_config) const {
  if (graph_config_path.empty()) {
    LOG(ERROR) << "graph_config_path cannot be empty.";
    return false;
  }
  return ensemble::load_graph_config_from_file(graph_config_path, graph_config);
}

bool EngineServer::select_node(const ensemble::GraphConfig& graph_config,
                               int32_t node_rank) {
  ensemble::NodeConfig node_config;
  if (!find_node_config(graph_config, node_rank, node_config)) {
    return false;
  }

  node_config_ = std::move(node_config);
  return true;
}

bool EngineServer::create_engine() {
  const std::string backend = options_.backend();
  if (backend == "vlm") {
    engine_ = std::make_unique<VLMEngine>(options_);
  } else if (backend == "dit") {
    engine_ = std::make_unique<DiTEngine>(options_);
  } else {
    LOG(ERROR) << "Unsupported engine backend for graph node "
               << node_config_.name << ": " << backend;
    return false;
  }
  return true;
}

bool EngineServer::init_engine_service(int32_t node_rank) {
  const bool is_leader = node_config_.ranks.begin()->first == node_rank;
  if (!is_leader) {
    LOG(INFO) << "Initialized non-leader engine runtime for graph node "
              << node_config_.name << ", node_rank=" << node_rank;
    return true;
  }

  service_ = std::make_unique<EngineService>(engine_.get());
  LOG(INFO) << "Initialized EngineService skeleton for graph node "
            << node_config_.name << ", node_rank=" << node_rank;
  if (!start_service_endpoint()) {
    return false;
  }
  if (!register_service()) {
    ServerRegistry::get_instance().unregister_server(server_name_);
    server_name_.clear();
    return false;
  }
  return true;
}

bool EngineServer::start_service_endpoint() {
  if (node_config_.endpoint.target.empty()) {
    LOG(ERROR) << "EngineService endpoint target cannot be empty for node: "
               << node_config_.name;
    return false;
  }

  const std::string server_name = "EngineService:" + node_config_.name;
  auto& registry = ServerRegistry::get_instance();
  if (registry.try_get_server(server_name) != nullptr) {
    LOG(ERROR) << "EngineService endpoint has already been registered: "
               << server_name;
    return false;
  }

  XllmServer* server = registry.register_server(server_name);
  if (!server->start(
          service_.get(), node_config_.endpoint.target, server_name)) {
    LOG(ERROR) << "Failed to start EngineService endpoint for graph node "
               << node_config_.name
               << ", target=" << node_config_.endpoint.target;
    registry.unregister_server(server_name);
    return false;
  }

  server_name_ = server_name;
  LOG(INFO) << "Started EngineService endpoint for graph node "
            << node_config_.name << ", target=" << node_config_.endpoint.target;
  return true;
}

bool EngineServer::register_service() {
  if (FLAGS_omni_master_addr.empty()) {
    LOG(ERROR) << "omni_master_addr cannot be empty.";
    return false;
  }

  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = kReadyRegisterTimeoutMs;
  options.max_retry = kReadyRegisterMaxRetry;

  brpc::Channel channel;
  if (channel.Init(FLAGS_omni_master_addr.c_str(),
                   /*load_balancer=*/"",
                   &options) != 0) {
    LOG(ERROR) << "Failed to initialize ready registration channel to "
               << FLAGS_omni_master_addr;
    return false;
  }

  proto::EnsembleNodeReadyRequest request;
  request.set_node_name(node_config_.name);
  request.set_target(node_config_.endpoint.target);

  proto::EnsembleNodeReadyResponse response;
  brpc::Controller controller;
  proto::EnsembleNodeReady_Stub stub(&channel);
  stub.RegisterReady(&controller, &request, &response, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "Failed to register ready for graph node "
               << node_config_.name << " to " << FLAGS_omni_master_addr << ": "
               << controller.ErrorText();
    return false;
  }
  if (!response.ok()) {
    LOG(ERROR) << "Ready registration rejected for graph node "
               << node_config_.name
               << ", target=" << node_config_.endpoint.target;
    return false;
  }

  LOG(INFO) << "Registered ready for graph node " << node_config_.name
            << ", target=" << node_config_.endpoint.target
            << ", omni_master_addr=" << FLAGS_omni_master_addr;
  return true;
}

}  // namespace xllm
