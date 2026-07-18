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

#include "core/distributed_runtime/engine_service_base.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "core/common/global_flags.h"
#include "core/distributed_runtime/node.h"
#include "core/util/threadpool.h"
#include "core/util/timer.h"

namespace xllm {

EngineServiceBase::EngineServiceBase(const NodeRuntimePlan& runtime_plan)
    : runtime_plan_(runtime_plan) {
  CHECK(!runtime_plan_.node_name.empty()) << "node name cannot be empty.";
  CHECK(!runtime_plan_.result_target.empty())
      << "result target cannot be empty.";
  CHECK_GT(runtime_plan_.timeout_ms, 0) << "node timeout must be positive.";

  if (runtime_plan_.downstream_endpoint.has_value()) {
    CHECK(runtime_plan_.downstream_node_name.has_value())
        << "downstream node name is required with downstream endpoint.";
    downstream_client_ =
        std::make_unique<Node>(*runtime_plan_.downstream_endpoint);
  }
  result_client_ = std::make_unique<Node>(runtime_plan_.result_target);

  const int32_t configured_threads = FLAGS_num_request_handling_threads;
  const size_t thread_count =
      static_cast<size_t>(std::max(configured_threads, 1));
  thread_pool_ = std::make_unique<ThreadPool>(thread_count);
}

EngineServiceBase::~EngineServiceBase() { stop(); }

EngineServiceExecutionResult EngineServiceBase::make_failure_result(
    NodeStatusCode code,
    std::string message) {
  EngineServiceExecutionResult result;
  result.status.code = code;
  result.status.message = std::move(message);
  return result;
}

void EngineServiceBase::Submit(::google::protobuf::RpcController* controller,
                               const proto::NodeData* request,
                               proto::Ack* response,
                               ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  (void)controller;

  Ack ack;
  if (request == nullptr) {
    ack.error_message = "NodeData request cannot be null.";
    ack.to_proto(response);
    return;
  }

  std::optional<NodeData> input = NodeData::from_proto(*request);
  if (!input.has_value()) {
    ack.error_message = "Failed to parse NodeData.";
    ack.to_proto(response);
    return;
  }

  thread_pool_->schedule([this, input = std::move(*input)]() mutable {
    EngineServiceExecutionResult result = execute_request(input);
    route_output(input.context, std::move(result));
  });
  ack.ok = true;
  ack.to_proto(response);
}

void EngineServiceBase::stop() { thread_pool_.reset(); }

EngineServiceExecutionResult EngineServiceBase::execute_request(
    const NodeData& input) {
  if (input.context.deadline_ms <= Timer::now_milliseconds()) {
    return make_failure_result(
        NodeStatusCode::TIMEOUT,
        "Request deadline expired before node execution.");
  }

  return execute(input);
}

void EngineServiceBase::route_output(const RequestContext& context,
                                     EngineServiceExecutionResult result) {
  if (result.status.code != NodeStatusCode::OK) {
    report_status(context, result.status.code, result.status.message);
    return;
  }
  if (!result.output.has_value()) {
    report_status(context,
                  NodeStatusCode::INTERNAL_ERROR,
                  "Successful node execution did not produce output.");
    return;
  }

  NodeData output = std::move(*result.output);
  output.context = context;
  output.source_node = runtime_plan_.node_name;

  if (runtime_plan_.final_output) {
    report_status(context, NodeStatusCode::OK, "", std::move(output.payload));
    return;
  }

  if (downstream_client_ == nullptr ||
      !runtime_plan_.downstream_node_name.has_value()) {
    report_status(context,
                  NodeStatusCode::INTERNAL_ERROR,
                  "Non-final node is missing downstream runtime plan.");
    return;
  }
  const int64_t remaining_ms = context.deadline_ms - Timer::now_milliseconds();
  if (remaining_ms <= 0) {
    report_status(context,
                  NodeStatusCode::TIMEOUT,
                  "Request deadline expired before downstream submission.");
    return;
  }

  output.target_node = *runtime_plan_.downstream_node_name;
  proto::NodeData proto_output;
  if (!output.to_proto(&proto_output)) {
    report_status(context,
                  NodeStatusCode::INTERNAL_ERROR,
                  "Failed to serialize downstream NodeData.");
    return;
  }

  proto::Ack proto_ack;
  brpc::Controller controller;
  controller.set_timeout_ms(static_cast<int>(remaining_ms));
  downstream_client_->submit(&controller, &proto_output, &proto_ack, nullptr);
  if (controller.Failed()) {
    report_status(context,
                  NodeStatusCode::DOWNSTREAM_FAILED,
                  "Downstream Submit RPC failed: " + controller.ErrorText());
    return;
  }
  const Ack ack = Ack::from_proto(proto_ack);
  if (!ack.ok) {
    report_status(context,
                  NodeStatusCode::DOWNSTREAM_FAILED,
                  ack.error_message.empty() ? "Downstream node rejected data."
                                            : ack.error_message);
  }
}

void EngineServiceBase::report_status(const RequestContext& context,
                                      NodeStatusCode code,
                                      const std::string& message,
                                      NodePayload payload) {
  NodeResult result;
  result.context = context;
  result.node_name = runtime_plan_.node_name;
  result.status.code = code;
  result.status.message = message;
  result.payload = std::move(payload);
  report_result(std::move(result));
}

void EngineServiceBase::report_result(NodeResult result) {
  proto::NodeResult proto_result;
  if (!result.to_proto(&proto_result)) {
    LOG(ERROR) << "Failed to serialize NodeResult for request "
               << result.context.request_id;
    return;
  }

  proto::Ack proto_ack;
  brpc::Controller controller;
  controller.set_timeout_ms(static_cast<int>(runtime_plan_.timeout_ms));
  result_client_->report(&controller, &proto_result, &proto_ack, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "Failed to report terminal result for request "
               << result.context.request_id << ": " << controller.ErrorText();
    return;
  }
  const Ack ack = Ack::from_proto(proto_ack);
  if (!ack.ok) {
    LOG(ERROR) << "Terminal result was rejected for request "
               << result.context.request_id << ": " << ack.error_message;
  }
}

}  // namespace xllm
