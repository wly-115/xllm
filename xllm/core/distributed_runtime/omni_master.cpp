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
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "core/distributed_runtime/ensemble_engine.h"
#include "core/distributed_runtime/ensemble_node_ready_service.h"
#include "core/distributed_runtime/ensemble_result_service.h"
#include "core/framework/ensemble/ensemble_protocol.h"
#include "core/framework/ensemble/node_payload.h"
#include "core/framework/ensemble/node_payload_fields.h"
#include "core/framework/request/omni_request.h"
#include "core/util/uuid.h"
#include "server/xllm_server_registry.h"

namespace xllm {
namespace {

std::string make_request_id() {
  thread_local ShortUUID short_uuid;
  return "omni-" + short_uuid.random();
}

void set_conversion_error(DiTRequestOutput& output, std::string message) {
  output.status = Status(StatusCode::UNKNOWN, std::move(message));
  output.finished = true;
}

std::optional<DiTGenerationOutput> parse_generation_metadata(
    const NodePayload& payload,
    std::string& error_message) {
  auto width_value = payload.get<std::string>(kPayloadFieldWidth);
  if (!width_value.has_value()) {
    error_message = payload.error_message();
    return std::nullopt;
  }

  auto height_value = payload.get<std::string>(kPayloadFieldHeight);
  if (!height_value.has_value()) {
    error_message = payload.error_message();
    return std::nullopt;
  }

  auto seed_value = payload.get<std::string>(kPayloadFieldSeed);
  if (!seed_value.has_value()) {
    error_message = payload.error_message();
    return std::nullopt;
  }

  DiTGenerationOutput output;
  try {
    output.width = static_cast<int32_t>(std::stol(*width_value));
  } catch (const std::exception&) {
    error_message = "Invalid image width field: " + *width_value;
    return std::nullopt;
  }
  try {
    output.height = static_cast<int32_t>(std::stol(*height_value));
  } catch (const std::exception&) {
    error_message = "Invalid image height field: " + *height_value;
    return std::nullopt;
  }
  try {
    output.seed = std::stoll(*seed_value);
  } catch (const std::exception&) {
    error_message = "Invalid image seed field: " + *seed_value;
    return std::nullopt;
  }
  return output;
}

bool convert_payload_to_dit_output(const NodePayload& payload,
                                   DiTRequestOutput* output) {
  CHECK(output != nullptr) << "dit request output cannot be null.";

  const std::string first_image_field = std::string(kPayloadFieldImage) + "_0";
  if (!payload.has_field(first_image_field)) {
    output->finished = true;
    return true;
  }

  std::string error_message;
  std::optional<DiTGenerationOutput> metadata =
      parse_generation_metadata(payload, error_message);
  if (!metadata.has_value()) {
    set_conversion_error(*output, std::move(error_message));
    return false;
  }

  int32_t index = 0;
  while (true) {
    const std::string image_field =
        std::string(kPayloadFieldImage) + "_" + std::to_string(index);
    if (!payload.has_field(image_field)) {
      break;
    }

    DiTGenerationOutput generation_output = *metadata;
    auto image_value = payload.get<NodePayloadBytes>(image_field);
    if (!image_value.has_value()) {
      set_conversion_error(*output, payload.error_message());
      return false;
    }
    generation_output.image.assign(image_value->begin(), image_value->end());
    output->outputs.emplace_back(std::move(generation_output));
    ++index;
  }
  output->finished = true;
  return true;
}

StatusCode convert_status_code(NodeStatusCode code) {
  switch (code) {
    case NodeStatusCode::TIMEOUT:
      return StatusCode::DEADLINE_EXCEEDED;
    case NodeStatusCode::DOWNSTREAM_FAILED:
      return StatusCode::UNAVAILABLE;
    case NodeStatusCode::EXECUTION_FAILED:
    case NodeStatusCode::INTERNAL_ERROR:
    case NodeStatusCode::UNSPECIFIED:
    case NodeStatusCode::OK:
      return StatusCode::UNKNOWN;
  }
  return StatusCode::UNKNOWN;
}

}  // namespace

OmniMaster::OmniMaster(const GraphConfig& graph_config,
                       std::string ready_target)
    : ready_timeout_ms_(graph_config.ready_timeout_ms),
      node_count_(static_cast<int32_t>(graph_config.nodes.size())),
      ready_service_(std::make_shared<EnsembleNodeReadyService>(node_count_)),
      ensemble_engine_(std::make_shared<EnsembleEngine>(graph_config)),
      result_service_(
          std::make_unique<EnsembleResultService>(ensemble_engine_)),
      ready_target_(std::move(ready_target)),
      result_target_(graph_config.result_target) {}

OmniMaster::~OmniMaster() {
  stop_ready_service();
  stop_result_service();
}

bool OmniMaster::complete_startup() {
  const bool ready = wait_nodes_ready();
  if (ready) {
    stop_ready_service();
  }
  return ready;
}

void OmniMaster::handle_request(OmniRequestInput input,
                                OmniRequestParams params,
                                DiTOutputCallback callback) {
  CHECK(callback != nullptr) << "dit output callback cannot be null.";
  const std::string request_id = make_request_id();
  OmniOutputFunc omni_callback = [request_id, callback = std::move(callback)](
                                     const NodeResult& result) mutable -> bool {
    DiTRequestOutput output;
    output.request_id = request_id;
    output.finished = true;
    if (result.status.code != NodeStatusCode::OK) {
      output.status = Status(convert_status_code(result.status.code),
                             result.status.message);
      return callback(std::move(output));
    }
    convert_payload_to_dit_output(result.payload, &output);
    return callback(std::move(output));
  };

  OmniRequestState state(
      std::move(input), std::move(params), std::move(omni_callback));
  std::shared_ptr<OmniRequest> request =
      std::make_shared<OmniRequest>(request_id, "", "", std::move(state));
  ensemble_engine_->submit(std::move(request));
}

bool OmniMaster::start_result_service() {
  if (result_target_.empty()) {
    LOG(ERROR) << "result endpoint target cannot be empty.";
    return false;
  }

  const std::string server_name = "EnsembleResult";
  ServerRegistry& registry = ServerRegistry::get_instance();
  if (registry.try_get_server(server_name) != nullptr) {
    LOG(ERROR) << "EnsembleResult server has already been registered.";
    return false;
  }

  XllmServer* result_server = registry.register_server(server_name);
  if (!result_server->start(
          result_service_.get(), result_target_, server_name)) {
    LOG(ERROR) << "Failed to start EnsembleResult server on address "
               << result_target_;
    registry.unregister_server(server_name);
    return false;
  }

  result_server_name_ = server_name;
  LOG(INFO) << "Started EnsembleResult server on address " << result_target_;
  return true;
}

bool OmniMaster::start_ready_service() {
  if (ready_target_.empty()) {
    LOG(ERROR) << "omni master ready target cannot be empty.";
    return false;
  }

  const std::string server_name = "EnsembleNodeReady";
  ServerRegistry& registry = ServerRegistry::get_instance();
  if (registry.try_get_server(server_name) != nullptr) {
    LOG(ERROR) << "EnsembleNodeReady server has already been registered.";
    return false;
  }

  XllmServer* ready_server = registry.register_server(server_name);
  if (!ready_server->start(ready_service_.get(), ready_target_, server_name)) {
    LOG(ERROR) << "Failed to start EnsembleNodeReady server on address "
               << ready_target_;
    registry.unregister_server(server_name);
    return false;
  }

  ready_server_name_ = server_name;
  LOG(INFO) << "Started EnsembleNodeReady server on address " << ready_target_;
  return true;
}

bool OmniMaster::wait_nodes_ready() {
  const std::unordered_set<std::string> ready_nodes =
      ready_service_->wait(ready_timeout_ms_);
  if (static_cast<int32_t>(ready_nodes.size()) != node_count_) {
    LOG(ERROR) << "Not all engine services are ready. expected=" << node_count_
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

void OmniMaster::stop_result_service() {
  if (result_server_name_.empty()) {
    return;
  }

  ServerRegistry::get_instance().unregister_server(result_server_name_);
  result_server_name_.clear();
}

}  // namespace xllm
