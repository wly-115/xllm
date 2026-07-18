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

#include <absl/time/time.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/distributed_runtime/ar_engine_service.h"
#include "core/distributed_runtime/dit_engine.h"
#include "core/distributed_runtime/dit_engine_service.h"
#include "core/distributed_runtime/engine_data_adapter.h"
#include "core/distributed_runtime/engine_service_base.h"
#include "core/distributed_runtime/qwen_image_edit_dit_adapter.h"
#include "core/distributed_runtime/qwen_vlm_encode_adapter.h"
#include "core/distributed_runtime/vlm_engine.h"
#include "core/framework/ensemble/engine_config.h"
#include "server/xllm_server_registry.h"
#include "worker.pb.h"

namespace xllm {
namespace {

constexpr int32_t kReadyRegisterTimeoutMs = 10000;
constexpr int32_t kReadyRegisterMaxRetry = 3;
constexpr int32_t kSchedulerStepTimeoutMs = 500;
constexpr const char* kEngineServiceServerPrefix = "EngineService:";
constexpr char kQwenVlmEncodeAdapter[] = "qwen_vlm_encode";
constexpr char kQwenImageEditDitAdapter[] = "qwen_image_edit_dit";

}  // namespace

EngineServer::EngineServer(const NodeRuntimePlan& runtime_plan)
    : runtime_plan_(runtime_plan) {
  engine_ = create_engine();
  CHECK(engine_ != nullptr);

  if (!runtime_plan_.is_leader) {
    return;
  }
  CHECK(engine_->init());
  CHECK(engine_->init_scheduler());
  start_scheduler();
  start_service();
}

EngineServer::~EngineServer() {
  stop_service_endpoint();
  service_.reset();
  stop_scheduler();
}

std::unique_ptr<Engine> EngineServer::create_engine() {
  const runtime::Options& options = runtime_plan_.runtime_options;
  const std::string& backend = runtime_plan_.backend;
  CHECK_EQ(options.backend(), backend)
      << "Runtime plan backend must match runtime options.";
  if (backend == "vlm") {
    return std::make_unique<VLMEngine>(options);
  } else if (backend == "dit") {
    return std::make_unique<DiTEngine>(options);
  }
  LOG(FATAL) << "Unsupported engine backend for graph node "
             << runtime_plan_.node_name << ": " << backend;
  return nullptr;
}

std::unique_ptr<EngineServiceBase> EngineServer::create_service() {
  if (runtime_plan_.backend == "vlm" &&
      runtime_plan_.adapter == kQwenVlmEncodeAdapter) {
    std::unique_ptr<ArEngineDataAdapter> adapter =
        std::make_unique<QwenVlmEncodeAdapter>(engine_->model_args());
    return std::make_unique<ArEngineService>(
        engine_.get(), std::move(adapter), runtime_plan_);
  }
  if (runtime_plan_.backend == "dit" &&
      runtime_plan_.adapter == kQwenImageEditDitAdapter) {
    DiTEngine* dit_engine = dynamic_cast<DiTEngine*>(engine_.get());
    CHECK(dit_engine != nullptr) << "DiT service requires DiTEngine.";
    std::unique_ptr<DitEngineDataAdapter> adapter =
        std::make_unique<QwenImageEditDitAdapter>();
    return std::make_unique<DitEngineService>(
        dit_engine, std::move(adapter), runtime_plan_);
  }
  LOG(FATAL) << "Unsupported backend and adapter combination for graph node "
             << runtime_plan_.node_name << ": backend=" << runtime_plan_.backend
             << ", adapter=" << runtime_plan_.adapter;
  return nullptr;
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
  service_ = create_service();
  CHECK(service_ != nullptr) << "Failed to create EngineService.";
  const std::string& target = runtime_plan_.service_target;
  CHECK(!target.empty()) << "EngineService endpoint target cannot be empty "
                         << "for node: " << runtime_plan_.node_name;

  const std::string server_name =
      std::string(kEngineServiceServerPrefix) + runtime_plan_.node_name;
  auto& registry = ServerRegistry::get_instance();
  CHECK(registry.try_get_server(server_name) == nullptr)
      << "EngineService endpoint has already been registered: " << server_name;

  XllmServer* server = registry.register_server(server_name);
  if (!server->start(service_.get(), target, server_name)) {
    registry.unregister_server(server_name);
    LOG(FATAL) << "Failed to start EngineService endpoint for graph node "
               << runtime_plan_.node_name << ", target=" << target;
  }

  service_server_name_ = server_name;
  register_ready();
}

void EngineServer::start_scheduler() {
  stop_scheduler_.store(false, std::memory_order_relaxed);
  scheduler_thread_ = std::thread([this]() {
    const absl::Duration timeout = absl::Milliseconds(kSchedulerStepTimeoutMs);
    while (!stop_scheduler_.load(std::memory_order_relaxed)) {
      engine_->step_scheduler(timeout);
    }
  });
}

void EngineServer::stop_scheduler() {
  stop_scheduler_.store(true, std::memory_order_relaxed);
  if (scheduler_thread_.joinable()) {
    scheduler_thread_.join();
  }
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
  request.set_node_name(runtime_plan_.node_name);

  proto::EnsembleNodeReadyResponse response;
  brpc::Controller controller;
  proto::EnsembleNodeReady_Stub stub(&channel);
  stub.RegisterReady(&controller, &request, &response, nullptr);
  CHECK(!controller.Failed())
      << "Failed to register ready for graph node " << runtime_plan_.node_name
      << " to " << FLAGS_omni_master_addr << ": " << controller.ErrorText();
  CHECK(response.ok()) << "Ready registration rejected for graph node "
                       << runtime_plan_.node_name;
}

void EngineServer::stop_service_endpoint() {
  if (service_server_name_.empty()) {
    return;
  }
  ServerRegistry::get_instance().unregister_server(service_server_name_);
  service_server_name_.clear();
}

}  // namespace xllm
