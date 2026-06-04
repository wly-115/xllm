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

#include "distributed_runtime/ensemble_node_ready_service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/closure_guard.h>
#include <glog/logging.h>

namespace xllm {

EnsembleNodeReadyService::EnsembleNodeReadyService(int32_t total_num)
    : total_num_(total_num) {
  CHECK_GE(total_num_, 0) << "total_num must be non-negative.";
}

bool EnsembleNodeReadyService::register_ready(const std::string& node_name) {
  if (node_name.empty()) {
    LOG(ERROR) << "node_name cannot be empty.";
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (ready_nodes_.find(node_name) != ready_nodes_.end()) {
    return true;
  }

  ready_nodes_.emplace(node_name);
  return true;
}

std::unordered_set<std::string> EnsembleNodeReadyService::wait(
    int64_t timeout_ms) {
  CHECK_GE(timeout_ms, 0) << "timeout_ms must be non-negative.";
  const absl::Time deadline = absl::Now() + absl::Milliseconds(timeout_ms);
  while (true) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (static_cast<int32_t>(ready_nodes_.size()) >= total_num_) {
        return ready_nodes_;
      }
    }
    if (absl::Now() >= deadline) {
      LOG(ERROR) << "Timeout waiting ensemble node ready.";
      std::lock_guard<std::mutex> lock(mutex_);
      return ready_nodes_;
    }
    absl::SleepFor(absl::Milliseconds(100));
  }
}

void EnsembleNodeReadyService::RegisterReady(
    ::google::protobuf::RpcController* controller,
    const proto::EnsembleNodeReadyRequest* request,
    proto::EnsembleNodeReadyResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  response->set_ok(register_ready(request->node_name()));
}

}  // namespace xllm
