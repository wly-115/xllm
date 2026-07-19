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
#include <memory>
#include <string>

#include "core/framework/ensemble/graph_config.h"
#include "core/framework/request/dit_request_output.h"
#include "core/framework/request/omni_request_state.h"

namespace xllm {

class EnsembleEngine;
class EnsembleNodeReadyService;
class EnsembleResultService;

class OmniMaster final {
 public:
  OmniMaster(const GraphConfig& graph_config, std::string ready_target);
  ~OmniMaster();

  bool start_result_service();
  bool start_ready_service();
  bool complete_startup();
  void handle_request(OmniRequestInput input,
                      OmniRequestParams params,
                      DiTOutputCallback callback);

 private:
  bool wait_nodes_ready();
  void stop_ready_service();
  void stop_result_service();

  int64_t ready_timeout_ms_ = 0;
  int32_t node_count_ = 0;
  std::shared_ptr<EnsembleNodeReadyService> ready_service_;
  std::shared_ptr<EnsembleEngine> ensemble_engine_;
  std::unique_ptr<EnsembleResultService> result_service_;
  std::string ready_target_;
  std::string result_target_;
  std::string ready_server_name_;
  std::string result_server_name_;
};

}  // namespace xllm
