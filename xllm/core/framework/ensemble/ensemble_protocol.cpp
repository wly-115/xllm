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

#include "core/framework/ensemble/ensemble_protocol.h"

#include <glog/logging.h>

#include <optional>
#include <string>
#include <utility>

#include "core/framework/request/worker_params_conversion.h"
#include "engine_service.pb.h"

namespace xllm {
namespace {

proto::NodeStatusCode node_status_code_to_proto(NodeStatusCode code) {
  return static_cast<proto::NodeStatusCode>(static_cast<int32_t>(code));
}

std::optional<NodeStatusCode> node_status_code_from_proto(
    proto::NodeStatusCode code) {
  switch (code) {
    case proto::NODE_STATUS_CODE_UNSPECIFIED:
      return NodeStatusCode::UNSPECIFIED;
    case proto::NODE_STATUS_CODE_OK:
      return NodeStatusCode::OK;
    case proto::NODE_STATUS_CODE_EXECUTION_FAILED:
      return NodeStatusCode::EXECUTION_FAILED;
    case proto::NODE_STATUS_CODE_TIMEOUT:
      return NodeStatusCode::TIMEOUT;
    case proto::NODE_STATUS_CODE_DOWNSTREAM_FAILED:
      return NodeStatusCode::DOWNSTREAM_FAILED;
    case proto::NODE_STATUS_CODE_INTERNAL_ERROR:
      return NodeStatusCode::INTERNAL_ERROR;
  }
  return std::nullopt;
}

}  // namespace

bool RequestContext::to_proto(proto::RequestContext* proto_context) const {
  if (proto_context == nullptr) {
    LOG(ERROR) << "RequestContext proto output cannot be null.";
    return false;
  }
  if (request_id.empty()) {
    LOG(ERROR) << "RequestContext request_id cannot be empty.";
    return false;
  }
  if (deadline_ms <= 0) {
    LOG(ERROR) << "RequestContext deadline_ms must be positive.";
    return false;
  }
  if (result_target.empty()) {
    LOG(ERROR) << "RequestContext result_target cannot be empty.";
    return false;
  }

  proto_context->Clear();
  proto_context->set_request_id(request_id);
  proto_context->set_deadline_ms(deadline_ms);
  proto_context->set_result_target(result_target);
  if (sampling_params.has_value()) {
    sampling_params_to_proto(*sampling_params,
                             proto_context->mutable_sampling_params());
  }
  if (dit_generation_params.has_value()) {
    dit_generation_params_to_proto(
        *dit_generation_params, proto_context->mutable_dit_generation_params());
  }
  return true;
}

std::optional<RequestContext> RequestContext::from_proto(
    const proto::RequestContext& proto_context) {
  if (proto_context.request_id().empty()) {
    LOG(ERROR) << "RequestContext request_id cannot be empty.";
    return std::nullopt;
  }
  if (proto_context.deadline_ms() <= 0) {
    LOG(ERROR) << "RequestContext deadline_ms must be positive.";
    return std::nullopt;
  }
  if (proto_context.result_target().empty()) {
    LOG(ERROR) << "RequestContext result_target cannot be empty.";
    return std::nullopt;
  }

  RequestContext context;
  context.request_id = proto_context.request_id();
  context.deadline_ms = proto_context.deadline_ms();
  context.result_target = proto_context.result_target();
  if (proto_context.has_sampling_params()) {
    RequestSamplingParam sampling_params;
    sampling_params_from_proto(proto_context.sampling_params(),
                               &sampling_params);
    context.sampling_params = std::move(sampling_params);
  }
  if (proto_context.has_dit_generation_params()) {
    DiTGenerationParams generation_params;
    dit_generation_params_from_proto(proto_context.dit_generation_params(),
                                     &generation_params);
    context.dit_generation_params = std::move(generation_params);
  }
  return context;
}

bool NodeData::to_proto(proto::NodeData* proto_data) const {
  if (proto_data == nullptr) {
    LOG(ERROR) << "NodeData proto output cannot be null.";
    return false;
  }
  if (target_node.empty()) {
    LOG(ERROR) << "NodeData target_node cannot be empty.";
    return false;
  }

  proto_data->Clear();
  if (!context.to_proto(proto_data->mutable_context())) {
    return false;
  }
  proto_data->set_source_node(source_node);
  proto_data->set_target_node(target_node);
  if (!payload.to_proto(proto_data->mutable_payload())) {
    LOG(ERROR) << "Failed to convert NodeData payload to proto: "
               << payload.error_message();
    return false;
  }
  return true;
}

std::optional<NodeData> NodeData::from_proto(
    const proto::NodeData& proto_data) {
  if (!proto_data.has_context()) {
    LOG(ERROR) << "NodeData context is missing.";
    return std::nullopt;
  }
  if (proto_data.target_node().empty()) {
    LOG(ERROR) << "NodeData target_node cannot be empty.";
    return std::nullopt;
  }

  std::optional<RequestContext> context =
      RequestContext::from_proto(proto_data.context());
  if (!context.has_value()) {
    return std::nullopt;
  }
  std::optional<NodePayload> payload =
      NodePayload::from_proto(proto_data.payload());
  if (!payload.has_value()) {
    return std::nullopt;
  }

  NodeData node_data;
  node_data.context = std::move(*context);
  node_data.source_node = proto_data.source_node();
  node_data.target_node = proto_data.target_node();
  node_data.payload = std::move(*payload);
  return node_data;
}

bool NodeStatus::to_proto(proto::NodeStatus* proto_status) const {
  if (proto_status == nullptr) {
    LOG(ERROR) << "NodeStatus proto output cannot be null.";
    return false;
  }
  if (code == NodeStatusCode::UNSPECIFIED) {
    LOG(ERROR) << "NodeStatus code cannot be unspecified.";
    return false;
  }
  if (code != NodeStatusCode::OK && message.empty()) {
    LOG(ERROR) << "Failed NodeStatus must include a message.";
    return false;
  }

  proto_status->Clear();
  proto_status->set_code(node_status_code_to_proto(code));
  proto_status->set_message(message);
  return true;
}

std::optional<NodeStatus> NodeStatus::from_proto(
    const proto::NodeStatus& proto_status) {
  std::optional<NodeStatusCode> code =
      node_status_code_from_proto(proto_status.code());
  if (!code.has_value() || *code == NodeStatusCode::UNSPECIFIED) {
    LOG(ERROR) << "NodeStatus code is invalid or unspecified.";
    return std::nullopt;
  }
  if (*code != NodeStatusCode::OK && proto_status.message().empty()) {
    LOG(ERROR) << "Failed NodeStatus must include a message.";
    return std::nullopt;
  }

  NodeStatus status;
  status.code = *code;
  status.message = proto_status.message();
  return status;
}

bool NodeResult::to_proto(proto::NodeResult* proto_result) const {
  if (proto_result == nullptr) {
    LOG(ERROR) << "NodeResult proto output cannot be null.";
    return false;
  }
  if (node_name.empty()) {
    LOG(ERROR) << "NodeResult node_name cannot be empty.";
    return false;
  }

  proto_result->Clear();
  if (!context.to_proto(proto_result->mutable_context())) {
    return false;
  }
  proto_result->set_node_name(node_name);
  if (!status.to_proto(proto_result->mutable_status())) {
    return false;
  }
  if (!payload.to_proto(proto_result->mutable_payload())) {
    LOG(ERROR) << "Failed to convert NodeResult payload to proto: "
               << payload.error_message();
    return false;
  }
  return true;
}

std::optional<NodeResult> NodeResult::from_proto(
    const proto::NodeResult& proto_result) {
  if (!proto_result.has_context()) {
    LOG(ERROR) << "NodeResult context is missing.";
    return std::nullopt;
  }
  if (proto_result.node_name().empty()) {
    LOG(ERROR) << "NodeResult node_name cannot be empty.";
    return std::nullopt;
  }
  if (!proto_result.has_status()) {
    LOG(ERROR) << "NodeResult status is missing.";
    return std::nullopt;
  }

  std::optional<RequestContext> context =
      RequestContext::from_proto(proto_result.context());
  if (!context.has_value()) {
    return std::nullopt;
  }
  std::optional<NodeStatus> status =
      NodeStatus::from_proto(proto_result.status());
  if (!status.has_value()) {
    return std::nullopt;
  }
  std::optional<NodePayload> payload =
      NodePayload::from_proto(proto_result.payload());
  if (!payload.has_value()) {
    return std::nullopt;
  }

  NodeResult node_result;
  node_result.context = std::move(*context);
  node_result.node_name = proto_result.node_name();
  node_result.status = std::move(*status);
  node_result.payload = std::move(*payload);
  return node_result;
}

void Ack::to_proto(proto::Ack* proto_ack) const {
  CHECK(proto_ack != nullptr) << "Ack proto output cannot be null.";
  proto_ack->Clear();
  proto_ack->set_ok(ok);
  proto_ack->set_error_message(error_message);
}

Ack Ack::from_proto(const proto::Ack& proto_ack) {
  Ack ack;
  ack.ok = proto_ack.ok();
  ack.error_message = proto_ack.error_message();
  return ack;
}

}  // namespace xllm
