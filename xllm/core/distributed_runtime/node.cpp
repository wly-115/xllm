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

#include "core/distributed_runtime/node.h"

#include <glog/logging.h>

#include <memory>

namespace xllm {

Node::Node(const std::string& endpoint) {
  brpc::ChannelOptions options;
  options.connection_type = "pooled";
  options.timeout_ms = -1;
  options.connect_timeout_ms = -1;
  options.max_retry = 3;

  if (channel_.Init(endpoint.c_str(), /*load_balancer_name=*/"", &options) !=
      0) {
    LOG(FATAL) << "Failed to initialize ensemble node channel: " << endpoint;
  }
  engine_stub_ = std::make_unique<proto::EngineService_Stub>(&channel_);
  result_stub_ = std::make_unique<proto::EnsembleResultService_Stub>(&channel_);
}

void Node::submit(brpc::Controller* cntl,
                  const proto::NodeData* input,
                  proto::Ack* output,
                  google::protobuf::Closure* done) {
  engine_stub_->Submit(cntl, input, output, done);
}

void Node::report(brpc::Controller* cntl,
                  const proto::NodeResult* input,
                  proto::Ack* output,
                  google::protobuf::Closure* done) {
  result_stub_->Report(cntl, input, output, done);
}

}  // namespace xllm
