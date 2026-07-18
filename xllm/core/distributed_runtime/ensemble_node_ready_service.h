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
#include <mutex>
#include <string>
#include <unordered_set>

#include "common/macros.h"
#include "worker.pb.h"

namespace xllm {

class EnsembleNodeReadyService final : public proto::EnsembleNodeReady {
 public:
  explicit EnsembleNodeReadyService(int32_t total_num);
  ~EnsembleNodeReadyService() override = default;

  std::unordered_set<std::string> wait(int64_t timeout_ms);

  void RegisterReady(::google::protobuf::RpcController* controller,
                     const proto::EnsembleNodeReadyRequest* request,
                     proto::EnsembleNodeReadyResponse* response,
                     ::google::protobuf::Closure* done) override;

 private:
  bool register_ready(const std::string& node_name);

  DISALLOW_COPY_AND_ASSIGN(EnsembleNodeReadyService);

  int32_t total_num_ = 0;
  mutable std::mutex mutex_;
  std::unordered_set<std::string> ready_nodes_;
};

}  // namespace xllm
