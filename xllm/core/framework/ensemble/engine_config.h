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

#include "core/framework/ensemble/graph_config.h"
#include "core/runtime/options.h"

namespace xllm {
namespace ensemble {

struct NodeLaunchConfig {
  std::string node_name;
  std::string service_target;
  runtime::Options runtime_options;
  int32_t graph_global_rank = 0;
  int32_t local_rank = 0;
  int32_t world_size = 1;
  bool is_leader = false;
};

NodeLaunchConfig resolve_node_launch_config(const GraphConfig& config,
                                            int32_t graph_global_rank);

}  // namespace ensemble
}  // namespace xllm
