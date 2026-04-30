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

#include <glog/logging.h>

#include <utility>

namespace xllm {

EngineService::EngineService(ensemble::NodeConfig node_config,
                             runtime::Options options,
                             Engine* engine)
    : node_config_(std::move(node_config)),
      options_(std::move(options)),
      engine_(engine) {
  CHECK(engine_ != nullptr) << "engine cannot be null.";
}

const std::string& EngineService::node_name() const {
  return node_config_.name;
}

const ensemble::EndpointConfig& EngineService::endpoint() const {
  return node_config_.endpoint;
}

const runtime::Options& EngineService::options() const { return options_; }

Engine* EngineService::engine() const { return engine_; }

}  // namespace xllm
