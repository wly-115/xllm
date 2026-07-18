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

#include "core/distributed_runtime/ensemble_result_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

#include <optional>
#include <string>
#include <utility>

#include "core/distributed_runtime/ensemble_engine.h"
#include "core/framework/ensemble/ensemble_protocol.h"

namespace xllm {

EnsembleResultService::EnsembleResultService(
    std::shared_ptr<EnsembleEngine> engine)
    : engine_(std::move(engine)) {
  CHECK(engine_ != nullptr) << "ensemble engine cannot be null.";
}

void EnsembleResultService::Report(
    ::google::protobuf::RpcController* controller,
    const proto::NodeResult* request,
    proto::Ack* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  (void)controller;

  Ack ack;
  if (request == nullptr) {
    ack.error_message = "NodeResult request cannot be null.";
    ack.to_proto(response);
    return;
  }

  std::optional<NodeResult> result = NodeResult::from_proto(*request);
  if (!result.has_value()) {
    const std::string request_id = request->has_context()
                                       ? request->context().request_id()
                                       : std::string();
    ack =
        engine_->handle_report_error(request_id, "Failed to parse NodeResult.");
    ack.to_proto(response);
    return;
  }

  engine_->handle_result(std::move(*result)).to_proto(response);
}

}  // namespace xllm
