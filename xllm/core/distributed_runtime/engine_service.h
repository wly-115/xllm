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

#include <string>

#include "common/macros.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/ensemble/graph_config.h"
#include "core/runtime/options.h"

namespace xllm {

class EngineService final {
 public:
  EngineService(ensemble::NodeConfig node_config,
                runtime::Options options,
                Engine* engine);

  const std::string& node_name() const;
  const ensemble::EndpointConfig& endpoint() const;
  const runtime::Options& options() const;
  Engine* engine() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(EngineService);

  ensemble::NodeConfig node_config_;
  runtime::Options options_;
  Engine* engine_ = nullptr;
};

}  // namespace xllm
