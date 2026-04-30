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

#include "core/distributed_runtime/engine.h"
#include "core/distributed_runtime/engine_service.h"
#include "core/framework/ensemble/graph_config.h"
#include "core/runtime/options.h"

namespace xllm {

class EngineServer final {
 public:
  EngineServer() = default;
  ~EngineServer();

  bool init(const std::string& graph_config_path, int32_t node_rank);
  bool init(const ensemble::GraphConfig& graph_config, int32_t node_rank);

  EngineService* service() const;
  Engine* engine() const;

 private:
  bool load_graph_config(const std::string& graph_config_path,
                         ensemble::GraphConfig& graph_config) const;
  bool select_node(const ensemble::GraphConfig& graph_config,
                   int32_t node_rank);
  bool create_engine();
  bool init_engine_service(int32_t node_rank);
  bool start_service_endpoint();
  bool register_service();

  ensemble::NodeConfig node_config_;
  runtime::Options options_;
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<EngineService> service_;
  std::string server_name_;
};

}  // namespace xllm
