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

#include "common/macros.h"
#include "engine_service.pb.h"

namespace xllm {

class EnsembleEngine;

class EnsembleResultService final : public proto::EnsembleResultService {
 public:
  explicit EnsembleResultService(std::shared_ptr<EnsembleEngine> engine);
  ~EnsembleResultService() override = default;

  void Report(::google::protobuf::RpcController* controller,
              const proto::NodeResult* request,
              proto::Ack* response,
              ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(EnsembleResultService);

  std::shared_ptr<EnsembleEngine> engine_;
};

}  // namespace xllm
