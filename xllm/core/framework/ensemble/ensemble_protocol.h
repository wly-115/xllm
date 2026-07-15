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

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/ensemble/node_payload.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace proto {
class Ack;
class NodeData;
class NodeResult;
class NodeStatus;
class RequestContext;
}  // namespace proto

class RequestContext final {
 public:
  std::string request_id;
  int64_t deadline_ms = 0;
  std::string result_target;
  std::optional<RequestSamplingParam> sampling_params;
  std::optional<DiTGenerationParams> dit_generation_params;

  bool to_proto(proto::RequestContext* proto_context) const;
  static std::optional<RequestContext> from_proto(
      const proto::RequestContext& proto_context);
};

class NodeData final {
 public:
  RequestContext context;
  std::string source_node;
  std::string target_node;
  NodePayload payload;

  bool to_proto(proto::NodeData* proto_data) const;
  static std::optional<NodeData> from_proto(const proto::NodeData& proto_data);
};

enum class NodeStatusCode : int32_t {
  UNSPECIFIED = 0,
  OK = 1,
  EXECUTION_FAILED = 3,
  TIMEOUT = 4,
  DOWNSTREAM_FAILED = 5,
  INTERNAL_ERROR = 6,
};

class NodeStatus final {
 public:
  NodeStatusCode code = NodeStatusCode::UNSPECIFIED;
  std::string message;

  bool to_proto(proto::NodeStatus* proto_status) const;
  static std::optional<NodeStatus> from_proto(
      const proto::NodeStatus& proto_status);
};

class NodeResult final {
 public:
  RequestContext context;
  std::string node_name;
  NodeStatus status;
  NodePayload payload;

  bool to_proto(proto::NodeResult* proto_result) const;
  static std::optional<NodeResult> from_proto(
      const proto::NodeResult& proto_result);
};

class Ack final {
 public:
  bool ok = false;
  std::string error_message;

  void to_proto(proto::Ack* proto_ack) const;
  static Ack from_proto(const proto::Ack& proto_ack);
};

}  // namespace xllm
