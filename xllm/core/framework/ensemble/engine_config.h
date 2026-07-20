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
#include <optional>
#include <string>

#include "core/framework/ensemble/graph_config.h"
#include "runtime/options.h"

namespace xllm {

struct NodeRuntimePlan {
  std::string node_name;
  std::string backend;
  std::string adapter;
  std::string service_target;
  std::string result_target;
  std::string ready_target;
  std::optional<std::string> downstream_node_name;
  std::optional<std::string> downstream_endpoint;
  bool final_output = false;
  int64_t timeout_ms = 30000;
  runtime::Options runtime_options;
  bool is_leader = false;
};

void apply_node_engine_config(const GraphConfig& config,
                              int32_t graph_global_rank);

NodeRuntimePlan build_node_runtime_plan(const GraphConfig& config,
                                        int32_t graph_global_rank,
                                        const std::string& ready_target);

}  // namespace xllm
