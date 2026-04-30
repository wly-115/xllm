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

#include "core/distributed_runtime/ensemble_engine.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

namespace xllm {

EnsembleEngine::EnsembleEngine(std::shared_ptr<const ensemble::Graph> graph)
    : graph_(std::move(graph)) {
  CHECK(graph_ != nullptr) << "graph cannot be null.";
}

}  // namespace xllm
