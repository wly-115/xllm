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

#include "core/distributed_runtime/ensemble_engine.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "core/distributed_runtime/node.h"
#include "core/framework/ensemble/graph_config.h"
#include "core/framework/ensemble/qwen_image_edit_preprocessor.h"
#include "core/util/timer.h"
#include "engine_service.pb.h"

namespace xllm {
namespace {

NodeResult make_failure_result(const RequestContext& context,
                               NodeStatusCode code,
                               const std::string& message) {
  NodeResult result;
  result.context = context;
  result.node_name = "ensemble_engine";
  result.status.code = code;
  result.status.message = message;
  return result;
}

void deliver_result(const std::shared_ptr<OmniRequest>& request,
                    const NodeResult& result) {
  CHECK(request != nullptr) << "omni request cannot be null.";
  const OmniOutputFunc& output_func = request->state().output_func();
  if (output_func && !output_func(result)) {
    LOG(WARNING) << "Omni output callback returned false, request_id="
                 << request->request_id();
  }
}

}  // namespace

EnsembleEngine::EnsembleEngine(const GraphConfig& config)
    : result_target_(config.result_target) {
  CHECK(!config.node_index.empty())
      << "graph config indices must be built before EnsembleEngine "
         "construction.";
  CHECK_EQ(config.get_root_nodes().size(), 1)
      << "EnsembleEngine requires exactly one root node.";
  CHECK(config.preprocessor_config.has_value())
      << "EnsembleEngine requires preprocessor config.";

  root_node_name_ = config.get_root_nodes().front();
  const NodeConfig* root_config = config.find_node(root_node_name_);
  CHECK(root_config != nullptr) << "Root node config is missing.";
  CHECK(!root_config->endpoint_target.empty())
      << "Root node endpoint target cannot be empty.";
  CHECK(!result_target_.empty()) << "Result endpoint target cannot be empty.";

  for (const NodeConfig& node : config.nodes) {
    CHECK_GT(node.timeout_ms, 0)
        << "Node timeout must be positive: " << node.name;
    request_timeout_ms_ += node.timeout_ms;
  }
  CHECK_GT(request_timeout_ms_, 0)
      << "Ensemble request timeout must be positive.";

  root_client_ = std::make_unique<Node>(root_config->endpoint_target);
  preprocessor_ =
      create_qwen_image_edit_preprocessor(*config.preprocessor_config);
  CHECK(preprocessor_ != nullptr)
      << "Failed to create qwen image edit preprocessor.";
}

EnsembleEngine::~EnsembleEngine() = default;

void EnsembleEngine::submit(std::shared_ptr<OmniRequest> request) {
  CHECK(request != nullptr) << "omni request cannot be null.";
  const RequestContext context = build_request_context(request);
  NodePayload payload;
  int64_t remaining_ms = 0;
  if (!preprocess_request(request, context, &payload, &remaining_ms)) {
    return;
  }

  proto::NodeData proto_data;
  if (!serialize_root_input(
          request, context, std::move(payload), &proto_data)) {
    return;
  }

  if (!track_active_request(request, context)) {
    return;
  }

  submit_to_root(context, remaining_ms, proto_data);
}

RequestContext EnsembleEngine::build_request_context(
    const std::shared_ptr<OmniRequest>& request) const {
  RequestContext context;
  context.request_id = request->request_id();
  context.deadline_ms = Timer::now_milliseconds() + request_timeout_ms_;
  context.result_target = result_target_;
  context.sampling_params = request->state().params().sampling_params;
  context.dit_generation_params =
      request->state().params().dit_generation_params;
  return context;
}

bool EnsembleEngine::preprocess_request(
    const std::shared_ptr<OmniRequest>& request,
    const RequestContext& context,
    NodePayload* payload,
    int64_t* remaining_ms) {
  CHECK(payload != nullptr) << "node payload cannot be null.";
  CHECK(remaining_ms != nullptr) << "remaining timeout cannot be null.";
  *payload = preprocessor_->preprocess(request->state().input());
  *remaining_ms = context.deadline_ms - Timer::now_milliseconds();
  if (*remaining_ms > 0) {
    return true;
  }
  deliver_result(request,
                 make_failure_result(context,
                                     NodeStatusCode::TIMEOUT,
                                     "Request deadline exceeded during "
                                     "preprocessing."));
  return false;
}

bool EnsembleEngine::serialize_root_input(
    const std::shared_ptr<OmniRequest>& request,
    const RequestContext& context,
    NodePayload payload,
    proto::NodeData* proto_data) {
  CHECK(proto_data != nullptr) << "node data proto cannot be null.";
  NodeData root_data;
  root_data.context = context;
  root_data.target_node = root_node_name_;
  root_data.payload = std::move(payload);
  if (root_data.to_proto(proto_data)) {
    return true;
  }
  deliver_result(request,
                 make_failure_result(context,
                                     NodeStatusCode::INTERNAL_ERROR,
                                     "Failed to serialize root NodeData."));
  return false;
}

bool EnsembleEngine::track_active_request(
    const std::shared_ptr<OmniRequest>& request,
    const RequestContext& context) {
  bool inserted = false;
  {
    std::lock_guard<std::mutex> lock(requests_mu_);
    inserted = requests_.emplace(context.request_id, request).second;
  }
  if (inserted) {
    return true;
  }
  LOG(ERROR) << "Duplicate active omni request ignored, request_id="
             << context.request_id;
  deliver_result(request,
                 make_failure_result(context,
                                     NodeStatusCode::INTERNAL_ERROR,
                                     "Duplicate active omni request id: " +
                                         context.request_id));
  return false;
}

void EnsembleEngine::submit_to_root(const RequestContext& context,
                                    int64_t remaining_ms,
                                    const proto::NodeData& proto_data) {
  proto::Ack proto_ack;
  brpc::Controller controller;
  controller.set_timeout_ms(static_cast<int>(remaining_ms));
  root_client_->submit(&controller, &proto_data, &proto_ack, nullptr);
  if (controller.Failed()) {
    abort_request(context.request_id,
                  make_failure_result(
                      context,
                      NodeStatusCode::DOWNSTREAM_FAILED,
                      "Root Submit RPC failed: " + controller.ErrorText()));
    return;
  }

  handle_root_ack(context, proto_ack);
}

void EnsembleEngine::handle_root_ack(const RequestContext& context,
                                     const proto::Ack& proto_ack) {
  const Ack ack = Ack::from_proto(proto_ack);
  if (ack.ok) {
    return;
  }
  const std::string error_message = ack.error_message.empty()
                                        ? "Root node rejected request."
                                        : ack.error_message;
  abort_request(context.request_id,
                make_failure_result(
                    context, NodeStatusCode::DOWNSTREAM_FAILED, error_message));
}

Ack EnsembleEngine::handle_result(NodeResult result) {
  Ack ack;
  const std::string request_id = result.context.request_id;
  if (request_id.empty()) {
    ack.error_message = "NodeResult request id cannot be empty.";
    LOG(ERROR) << ack.error_message;
    return ack;
  }

  std::shared_ptr<OmniRequest> request = remove_request(request_id);
  if (request == nullptr) {
    ack.error_message = "Unknown or completed request id: " + request_id;
    LOG(WARNING) << ack.error_message;
    return ack;
  }

  deliver_result(request, result);
  ack.ok = true;
  return ack;
}

Ack EnsembleEngine::handle_report_error(const std::string& request_id,
                                        const std::string& error_message) {
  Ack ack;
  if (request_id.empty()) {
    ack.error_message = error_message;
    LOG(ERROR) << "Cannot abort malformed result without request id: "
               << error_message;
    return ack;
  }

  RequestContext context;
  context.request_id = request_id;
  abort_request(request_id,
                make_failure_result(context,
                                    NodeStatusCode::INTERNAL_ERROR,
                                    "Invalid node result: " + error_message));
  ack.error_message = error_message;
  return ack;
}

std::shared_ptr<OmniRequest> EnsembleEngine::remove_request(
    const std::string& request_id) {
  std::lock_guard<std::mutex> lock(requests_mu_);
  auto request_it = requests_.find(request_id);
  if (request_it == requests_.end()) {
    return nullptr;
  }
  std::shared_ptr<OmniRequest> request = std::move(request_it->second);
  requests_.erase(request_it);
  return request;
}

void EnsembleEngine::abort_request(const std::string& request_id,
                                   NodeResult result) {
  std::shared_ptr<OmniRequest> request = remove_request(request_id);
  if (request == nullptr) {
    LOG(ERROR) << "Cannot abort unknown omni request, request_id="
               << request_id;
    return;
  }
  deliver_result(request, result);
}

}  // namespace xllm
