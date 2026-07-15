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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "engine_service.pb.h"

namespace xllm {

using NodePayloadBytes = std::vector<uint8_t>;
using NodePayloadValue = std::variant<std::string,
                                      NodePayloadBytes,
                                      torch::Tensor,
                                      MMData,
                                      std::vector<int32_t>>;

class NodePayload final {
 public:
  NodePayload() = default;

  static std::optional<NodePayload> from_proto(
      const proto::NodePayload& payload);

  bool to_proto(proto::NodePayload* payload) const;

  bool has_field(const std::string& name) const;
  bool set(const std::string& name, std::string value);
  bool set(const std::string& name, NodePayloadBytes value);
  bool set(const std::string& name, const torch::Tensor& value);
  bool set(const std::string& name, const MMData& value);
  bool set(const std::string& name, const std::vector<int32_t>& value);

  template <typename T>
  std::optional<T> get(const std::string& name) const {
    const NodePayloadValue* field = find_field(name);
    if (field == nullptr) {
      return std::nullopt;
    }
    const auto* typed_value = std::get_if<T>(field);
    if (typed_value == nullptr) {
      fail("Field type mismatch: " + name);
      return std::nullopt;
    }
    return *typed_value;
  }

  const std::string& error_message() const { return error_message_; }

 private:
  bool set_value(const std::string& name, NodePayloadValue value);
  const NodePayloadValue* find_field(const std::string& name) const;
  bool fail(std::string error_message) const;

  std::unordered_map<std::string, NodePayloadValue> fields_;
  mutable std::string error_message_;
};

}  // namespace xllm
