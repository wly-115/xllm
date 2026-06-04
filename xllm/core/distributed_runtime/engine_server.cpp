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

EngineServer::EngineServer() = default;

EngineServer::~EngineServer() { stop_service_endpoint(); }

void EngineServer::init(const ensemble::NodeLaunchConfig& launch_config) {
  launch_config_ = launch_config;
  engine_ = create_engine();
  CHECK(engine_ != nullptr);

  if (!launch_config_.is_leader) {
    return;
  }

  start_service();
}

std::unique_ptr<Engine> EngineServer::create_engine() {
  const runtime::Options& options = launch_config_.runtime_options;
  const std::string backend = options.backend();
  if (backend == "vlm") {
    return std::make_unique<VLMEngine>(options);
  }
  if (backend == "dit") {
    return std::make_unique<DiTEngine>(options);
  }
  LOG(FATAL) << "Unsupported engine backend for graph node "
             << launch_config_.node_name << ": " << backend;
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

void EngineServer::start_service() {
  CHECK(engine_ != nullptr);
  CHECK(engine_->init());
  service_ = std::make_unique<EngineService>(engine_.get());
  const std::string& target = launch_config_.service_target;
  CHECK(!target.empty()) << "EngineService endpoint target cannot be empty "
                         << "for node: " << launch_config_.node_name;

  const std::string server_name =
      std::string(kEngineServiceServerPrefix) + launch_config_.node_name;
  auto& registry = ServerRegistry::get_instance();
  CHECK(registry.try_get_server(server_name) == nullptr)
      << "EngineService endpoint has already been registered: " << server_name;

  XllmServer* server = registry.register_server(server_name);
  if (!server->start(service_.get(), target, server_name)) {
    registry.unregister_server(server_name);
    LOG(FATAL) << "Failed to start EngineService endpoint for graph node "
               << launch_config_.node_name << ", target=" << target;
  }

  service_server_name_ = server_name;
  register_ready();
}

void EngineServer::register_ready() {
  CHECK(!FLAGS_omni_master_addr.empty()) << "omni_master_addr cannot be empty.";

  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = kReadyRegisterTimeoutMs;
  options.max_retry = kReadyRegisterMaxRetry;

  brpc::Channel channel;
  CHECK_EQ(channel.Init(FLAGS_omni_master_addr.c_str(),
                        /*load_balancer=*/"",
                        &options),
           0)
      << "Failed to initialize ready registration channel to "
      << FLAGS_omni_master_addr;

  proto::EnsembleNodeReadyRequest request;
  request.set_node_name(launch_config_.node_name);

  proto::EnsembleNodeReadyResponse response;
  brpc::Controller controller;
  proto::EnsembleNodeReady_Stub stub(&channel);
  stub.RegisterReady(&controller, &request, &response, nullptr);
  CHECK(!controller.Failed())
      << "Failed to register ready for graph node " << launch_config_.node_name
      << " to " << FLAGS_omni_master_addr << ": " << controller.ErrorText();
  CHECK(response.ok()) << "Ready registration rejected for graph node "
                       << launch_config_.node_name;
}

void EngineServer::stop_service_endpoint() {
  if (service_server_name_.empty()) {
    return;
  }
  ServerRegistry::get_instance().unregister_server(service_server_name_);
  service_server_name_.clear();
}

}  // namespace xllm
