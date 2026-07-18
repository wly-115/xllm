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

#include <memory>
#include <optional>
#include <string>

#include "common/macros.h"
#include "core/framework/ensemble/engine_config.h"
#include "core/framework/ensemble/ensemble_protocol.h"
#include "engine_service.pb.h"

namespace xllm {

class Node;
class ThreadPool;

struct EngineServiceExecutionResult {
  NodeStatus status;
  std::optional<NodeData> output;
};

class EngineServiceBase : public proto::EngineService {
 public:
  explicit EngineServiceBase(const NodeRuntimePlan& runtime_plan);
  ~EngineServiceBase() override;

  void Submit(::google::protobuf::RpcController* controller,
              const proto::NodeData* request,
              proto::Ack* response,
              ::google::protobuf::Closure* done) final;

 protected:
  static EngineServiceExecutionResult make_failure_result(NodeStatusCode code,
                                                          std::string message);
  void stop();
  virtual EngineServiceExecutionResult execute(const NodeData& input) = 0;

 private:
  EngineServiceExecutionResult execute_request(const NodeData& input);
  void route_output(const RequestContext& context,
                    EngineServiceExecutionResult result);
  void report_status(const RequestContext& context,
                     NodeStatusCode code,
                     const std::string& message,
                     NodePayload payload = NodePayload());
  void report_result(NodeResult result);

  DISALLOW_COPY_AND_ASSIGN(EngineServiceBase);

  NodeRuntimePlan runtime_plan_;
  std::unique_ptr<Node> downstream_client_;
  std::unique_ptr<Node> result_client_;
  std::unique_ptr<ThreadPool> thread_pool_;
};

}  // namespace xllm
