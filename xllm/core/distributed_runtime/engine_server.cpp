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

#include "core/common/global_flags.h"
#include "core/distributed_runtime/dit_engine.h"
#include "core/distributed_runtime/engine_service.h"
#include "core/distributed_runtime/vlm_engine.h"
#include "core/framework/ensemble/engine_config.h"
#include "server/xllm_server_registry.h"
#include "worker.pb.h"

namespace xllm {
namespace {

constexpr int32_t kReadyRegisterTimeoutMs = 10000;
constexpr int32_t kReadyRegisterMaxRetry = 3;
constexpr const char* kEngineServiceServerPrefix = "EngineService:";

}  // namespace

EngineServer::~EngineServer() { stop_service_endpoint(); }

bool EngineServer::init(const ensemble::GraphConfig& graph_config,
                        int32_t node_rank) {
  for (const ensemble::NodeConfig& candidate : graph_config.nodes) {
    if (candidate.ranks.find(node_rank) != candidate.ranks.end()) {
      node_config_ = candidate;
      break;
    }
  }
  ensemble::initialize_engine_options_from_config(
      graph_config, node_rank, options_);
  CHECK(create_engine());

  if (node_config_.ranks.begin()->first != node_rank) {
    LOG(INFO) << "Initialized non-leader engine runtime for graph node "
              << node_config_.name << ", node_rank=" << node_rank;
    return true;
  }

  return start_leader_service(node_rank);
}

void EngineServer::create_engine() {
  const std::string backend = options_.backend();
  if (backend == "vlm") {
    engine_ = std::make_unique<VLMEngine>(options_);
  } else if (backend == "dit") {
    engine_ = std::make_unique<DiTEngine>(options_);
  } else {
    LOG(FATAL) << "Unsupported engine backend for graph node "
               << node_config_.name << ": " << backend;
  }
}

bool EngineServer::exposes_service() const {
  return !service_server_name_.empty();
}

void EngineServer::run() {
  CHECK(!service_server_name_.empty())
      << "EngineServer does not expose EngineService.";

  XllmServer* server =
      ServerRegistry::get_instance().try_get_server(service_server_name_);
  CHECK(server != nullptr) << "EngineService server is not registered: "
                           << service_server_name_;
  server->run();
}

bool EngineServer::start_leader_service(int32_t node_rank) {
  CHECK(engine_->init());

  service_ = std::make_unique<EngineService>(engine_.get());
  LOG(INFO) << "Initialized EngineService skeleton for graph node "
            << node_config_.name << ", node_rank=" << node_rank;
  const std::string& target = node_config_.endpoint.target;
  if (target.empty()) {
    LOG(ERROR) << "EngineService endpoint target cannot be empty for node: "
               << node_config_.name;
    return false;
  }

  const std::string server_name =
      std::string(kEngineServiceServerPrefix) + node_config_.name;
  auto& registry = ServerRegistry::get_instance();
  if (registry.try_get_server(server_name) != nullptr) {
    LOG(ERROR) << "EngineService endpoint has already been registered: "
               << server_name;
    return false;
  }

  XllmServer* server = registry.register_server(server_name);
  if (!server->start(service_.get(), target, server_name)) {
    LOG(ERROR) << "Failed to start EngineService endpoint for graph node "
               << node_config_.name << ", target=" << target;
    registry.unregister_server(server_name);
    return false;
  }

  service_server_name_ = server_name;
  LOG(INFO) << "Started EngineService endpoint for graph node "
            << node_config_.name << ", target=" << target;
  if (!register_ready()) {
    stop_service_endpoint();
    return false;
  }
  return true;
}

bool EngineServer::register_ready() {
  if (FLAGS_omni_master_addr.empty()) {
    LOG(ERROR) << "omni_master_addr cannot be empty.";
    return false;
  }
  const std::string& target = node_config_.endpoint.target;

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
  request.set_target(target);

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
               << node_config_.name << ", target=" << target;
    return false;
  }

  LOG(INFO) << "Registered ready for graph node " << node_config_.name
            << ", target=" << target
            << ", omni_master_addr=" << FLAGS_omni_master_addr;
  return true;
}

void EngineServer::stop_service_endpoint() {
  if (service_server_name_.empty()) {
    return;
  }
  ServerRegistry::get_instance().unregister_server(service_server_name_);
  service_server_name_.clear();
}

}  // namespace xllm
