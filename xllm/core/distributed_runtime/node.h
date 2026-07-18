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

#include <brpc/channel.h>
#include <brpc/controller.h>

#include <memory>
#include <string>

#include "engine_service.pb.h"

namespace xllm {

class Node final {
 public:
  explicit Node(const std::string& endpoint);

  void submit(brpc::Controller* cntl,
              const proto::NodeData* input,
              proto::Ack* output,
              google::protobuf::Closure* done);

  void report(brpc::Controller* cntl,
              const proto::NodeResult* input,
              proto::Ack* output,
              google::protobuf::Closure* done);

 private:
  brpc::Channel channel_;
  std::unique_ptr<proto::EngineService_Stub> engine_stub_;
  std::unique_ptr<proto::EnsembleResultService_Stub> result_stub_;
};

}  // namespace xllm
