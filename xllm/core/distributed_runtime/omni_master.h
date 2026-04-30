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

namespace xllm {

namespace ensemble {
class Graph;
}  // namespace ensemble

class EnsembleEngine;
class EnsembleNodeReadyService;

class OmniMaster final {
 public:
  OmniMaster() = default;

  bool init(const std::string& graph_config_path, int32_t node_rank);

 private:
  bool validate_init_args(const std::string& graph_config_path,
                          int32_t node_rank) const;
  bool load_graph_config(const std::string& graph_config_path);
  bool wait_engine_services_ready();
  std::shared_ptr<const ensemble::Graph> build_graph() const;
  bool create_ensemble_engine(std::shared_ptr<const ensemble::Graph> graph);

  ensemble::GraphConfig graph_config_;
  std::shared_ptr<EnsembleNodeReadyService> ready_service_;
  std::shared_ptr<EnsembleEngine> ensemble_engine_;
};

}  // namespace xllm
