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
#include <mutex>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "core/framework/ensemble/ensemble_protocol.h"
#include "core/framework/request/omni_request.h"

namespace xllm {

class GraphConfig;
class Node;
class QwenImageEditPreprocessor;

class EnsembleEngine final {
 public:
  explicit EnsembleEngine(const GraphConfig& config);
  ~EnsembleEngine();

  void submit(std::shared_ptr<OmniRequest> request);
  Ack handle_result(NodeResult result);
  Ack handle_report_error(const std::string& request_id,
                          const std::string& error_message);

 private:
  RequestContext build_request_context(
      const std::shared_ptr<OmniRequest>& request) const;
  bool preprocess_request(const std::shared_ptr<OmniRequest>& request,
                          const RequestContext& context,
                          NodePayload* payload,
                          int64_t* remaining_ms);
  bool serialize_root_input(const std::shared_ptr<OmniRequest>& request,
                            const RequestContext& context,
                            NodePayload payload,
                            proto::NodeData* proto_data);
  bool track_active_request(const std::shared_ptr<OmniRequest>& request,
                            const RequestContext& context);
  void submit_to_root(const RequestContext& context,
                      int64_t remaining_ms,
                      const proto::NodeData& proto_data);
  void handle_root_ack(const RequestContext& context,
                       const proto::Ack& proto_ack);
  std::shared_ptr<OmniRequest> remove_request(const std::string& request_id);
  void abort_request(const std::string& request_id, NodeResult result);

  std::string root_node_name_;
  std::string result_target_;
  int64_t request_timeout_ms_ = 0;
  std::unique_ptr<Node> root_client_;
  std::unique_ptr<QwenImageEditPreprocessor> preprocessor_;
  std::mutex requests_mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<OmniRequest>> requests_;
};

}  // namespace xllm
