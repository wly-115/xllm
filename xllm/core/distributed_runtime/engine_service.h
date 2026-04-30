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

#include "common/macros.h"
#include "core/distributed_runtime/engine.h"
#include "engine_service.pb.h"

namespace xllm {

class EngineService final : public proto::EngineService {
 public:
  explicit EngineService(Engine* engine);
  ~EngineService() override = default;

  void Submit(::google::protobuf::RpcController* controller,
              const proto::EngineServiceRequest* request,
              proto::EngineServiceResponse* response,
              ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(EngineService);

  Engine* engine_ = nullptr;
};

}  // namespace xllm
