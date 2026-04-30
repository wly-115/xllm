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

#include "core/distributed_runtime/engine_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

namespace xllm {

EngineService::EngineService(Engine* engine) : engine_(engine) {
  CHECK(engine_ != nullptr) << "engine cannot be null.";
}

void EngineService::Submit(::google::protobuf::RpcController* controller,
                           const proto::EngineServiceRequest* request,
                           proto::EngineServiceResponse* response,
                           ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  (void)controller;
  (void)request;
  response->set_ok(false);
  response->set_error_message("EngineService Submit is not implemented.");
}

}  // namespace xllm
