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

#include "core/framework/ensemble/node_payload.h"

#include <glog/logging.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/util/utils.h"

namespace xllm {
namespace {

bool mm_value_to_proto(const MMValue& value, proto::MMValue* proto_value) {
  if (proto_value == nullptr) {
    LOG(ERROR) << "MMValue proto output cannot be null.";
    return false;
  }

  if (const auto* tensor = std::get_if<torch::Tensor>(&value)) {
    return util::torch_to_proto(tensor->cpu().contiguous(),
                                proto_value->mutable_single_tensor());
  }

  const auto* tensors = std::get_if<std::vector<torch::Tensor>>(&value);
  if (tensors == nullptr) {
    LOG(ERROR) << "Unsupported MMValue type.";
    return false;
  }
  proto::TensorList* tensor_list = proto_value->mutable_tensor_list();
  for (const torch::Tensor& tensor : *tensors) {
    if (!util::torch_to_proto(tensor.cpu().contiguous(),
                              tensor_list->add_tensors())) {
      return false;
    }
  }
  return true;
}

std::optional<MMValue> proto_to_mm_value(const proto::MMValue& proto_value) {
  if (proto_value.has_single_tensor()) {
    torch::Tensor tensor = util::proto_to_torch(proto_value.single_tensor());
    if (!tensor.defined()) {
      return std::nullopt;
    }
    return MMValue(std::move(tensor));
  }

  if (!proto_value.has_tensor_list()) {
    LOG(ERROR) << "MMValue proto has no value.";
    return std::nullopt;
  }
  std::vector<torch::Tensor> tensors;
  tensors.reserve(proto_value.tensor_list().tensors_size());
  for (const proto::Tensor& proto_tensor :
       proto_value.tensor_list().tensors()) {
    torch::Tensor tensor = util::proto_to_torch(proto_tensor);
    if (!tensor.defined()) {
      return std::nullopt;
    }
    tensors.emplace_back(std::move(tensor));
  }
  return MMValue(std::move(tensors));
}

bool mm_dict_to_proto(
    const MMDict& dict,
    google::protobuf::Map<std::string, proto::MMValue>* proto_dict) {
  for (const auto& [key, value] : dict) {
    if (!mm_value_to_proto(value, &(*proto_dict)[key])) {
      LOG(ERROR) << "Failed to convert MMValue to proto: " << key;
      return false;
    }
  }
  return true;
}

std::optional<MMDict> proto_to_mm_dict(
    const google::protobuf::Map<std::string, proto::MMValue>& proto_dict) {
  MMDict dict;
  for (const auto& [key, proto_value] : proto_dict) {
    std::optional<MMValue> value = proto_to_mm_value(proto_value);
    if (!value.has_value()) {
      LOG(ERROR) << "Failed to convert MMValue from proto: " << key;
      return std::nullopt;
    }
    dict.emplace(key, std::move(value.value()));
  }
  return dict;
}

bool mm_data_to_proto(const MMData& mm_data, proto::MMData* proto_mm_data) {
  if (!mm_data.valid()) {
    LOG(ERROR) << "MMData field must be valid.";
    return false;
  }
  if (proto_mm_data == nullptr) {
    LOG(ERROR) << "MMData proto output cannot be null.";
    return false;
  }

  proto_mm_data->Clear();
  proto_mm_data->set_type(mm_data.type());
  if (mm_data.hold<MMDict>()) {
    return mm_dict_to_proto(mm_data.items<MMDict>(),
                            proto_mm_data->mutable_dict());
  }

  for (const MMDataItem& item : mm_data.items<MMItemVec>()) {
    proto::MMDataItem* proto_item = proto_mm_data->add_items();
    proto_item->set_type(item.type());
    if (!mm_dict_to_proto(item.data(), proto_item->mutable_dict())) {
      return false;
    }
    const MMItemState::TokenPos& token_pos = item.state().token_pos();
    proto_item->set_token_offset(token_pos.offset);
    proto_item->set_token_length(token_pos.length);
  }
  return true;
}

bool proto_to_mm_data(const proto::MMData& proto_mm_data, MMData* mm_data) {
  if (mm_data == nullptr) {
    LOG(ERROR) << "MMData output cannot be null.";
    return false;
  }
  if (proto_mm_data.type() == MMType::NONE) {
    LOG(ERROR) << "MMData proto type cannot be NONE.";
    return false;
  }

  if (proto_mm_data.items_size() == 0) {
    std::optional<MMDict> dict = proto_to_mm_dict(proto_mm_data.dict());
    if (!dict.has_value()) {
      return false;
    }
    *mm_data = MMData(proto_mm_data.type(), dict.value());
    return true;
  }

  MMItemVec items;
  items.reserve(proto_mm_data.items_size());
  for (const proto::MMDataItem& proto_item : proto_mm_data.items()) {
    std::optional<MMDict> dict = proto_to_mm_dict(proto_item.dict());
    if (!dict.has_value()) {
      return false;
    }
    MMDataItem item(static_cast<MMType::Value>(proto_item.type()),
                    dict.value());
    item.mutable_state().mutable_token_pos() = {proto_item.token_offset(),
                                                proto_item.token_length()};
    items.emplace_back(std::move(item));
  }
  *mm_data = MMData(proto_mm_data.type(), items);
  return true;
}

}  // namespace

std::optional<NodePayload> NodePayload::from_proto(
    const proto::NodePayload& payload) {
  NodePayload node_payload;
  for (const auto& field : payload.fields()) {
    const std::string& name = field.first;
    const proto::PayloadValue& value = field.second;
    switch (value.kind_case()) {
      case proto::PayloadValue::kStringValue:
        if (!node_payload.set(name, value.string_value())) {
          break;
        }
        continue;
      case proto::PayloadValue::kBytesValue: {
        const std::string& bytes = value.bytes_value();
        NodePayloadBytes binary(bytes.begin(), bytes.end());
        if (!node_payload.set(name, std::move(binary))) {
          break;
        }
        continue;
      }
      case proto::PayloadValue::kTensorValue: {
        torch::Tensor tensor = util::proto_to_torch(value.tensor_value());
        if (!tensor.defined()) {
          node_payload.fail("Failed to convert tensor field from proto: " +
                            name);
          break;
        }
        if (!node_payload.set(name, tensor)) {
          break;
        }
        continue;
      }
      case proto::PayloadValue::kMmDataValue: {
        MMData mm_data;
        if (!proto_to_mm_data(value.mm_data_value(), &mm_data)) {
          node_payload.fail("Failed to convert MMData field from proto: " +
                            name);
          break;
        }
        if (!node_payload.set(name, mm_data)) {
          break;
        }
        continue;
      }
      case proto::PayloadValue::kIntListValue: {
        std::vector<int32_t> values(value.int_list_value().values().begin(),
                                    value.int_list_value().values().end());
        if (!node_payload.set(name, values)) {
          break;
        }
        continue;
      }
      case proto::PayloadValue::KIND_NOT_SET:
        node_payload.fail("Payload field kind is unset: " + name);
        break;
    }

    return std::nullopt;
  }
  return node_payload;
}

bool NodePayload::to_proto(proto::NodePayload* payload) const {
  if (payload == nullptr) {
    return fail("NodePayload proto output cannot be null.");
  }

  payload->Clear();
  for (const auto& field : fields_) {
    const std::string& name = field.first;
    const NodePayloadValue& value = field.second;
    proto::PayloadValue& proto_value = (*payload->mutable_fields())[name];

    if (const auto* string_value = std::get_if<std::string>(&value)) {
      proto_value.set_string_value(*string_value);
      continue;
    }
    if (const auto* bytes_value = std::get_if<NodePayloadBytes>(&value)) {
      proto_value.set_bytes_value(bytes_value->data(), bytes_value->size());
      continue;
    }
    if (const auto* tensor_value = std::get_if<torch::Tensor>(&value)) {
      if (!util::torch_to_proto(tensor_value->cpu().contiguous(),
                                proto_value.mutable_tensor_value())) {
        return fail("Failed to convert tensor field to proto: " + name);
      }
      continue;
    }
    if (const auto* mm_data_value = std::get_if<MMData>(&value)) {
      if (!mm_data_to_proto(*mm_data_value,
                            proto_value.mutable_mm_data_value())) {
        return fail("Failed to convert MMData field to proto: " + name);
      }
      continue;
    }
    if (const auto* int_list_value =
            std::get_if<std::vector<int32_t>>(&value)) {
      proto::IntList* proto_int_list = proto_value.mutable_int_list_value();
      for (int32_t item : *int_list_value) {
        proto_int_list->add_values(item);
      }
      continue;
    }
  }
  return true;
}

bool NodePayload::has_field(const std::string& name) const {
  return fields_.find(name) != fields_.end();
}

bool NodePayload::set(const std::string& name, std::string value) {
  return set_value(name, std::move(value));
}

bool NodePayload::set(const std::string& name, NodePayloadBytes value) {
  return set_value(name, std::move(value));
}

bool NodePayload::set(const std::string& name, const torch::Tensor& value) {
  if (!value.defined()) {
    return fail("Tensor field must be defined: " + name);
  }
  return set_value(name, value);
}

bool NodePayload::set(const std::string& name, const MMData& value) {
  if (!value.valid()) {
    return fail("MMData field must be valid: " + name);
  }
  return set_value(name, value);
}

bool NodePayload::set(const std::string& name,
                      const std::vector<int32_t>& value) {
  return set_value(name, value);
}

bool NodePayload::set_value(const std::string& name, NodePayloadValue value) {
  if (name.empty()) {
    return fail("Payload field name cannot be empty.");
  }
  auto insert_result = fields_.emplace(name, std::move(value));
  if (!insert_result.second) {
    return fail("Duplicated payload field: " + name);
  }
  return true;
}

const NodePayloadValue* NodePayload::find_field(const std::string& name) const {
  auto field_it = fields_.find(name);
  if (field_it == fields_.end()) {
    fail("Missing payload field: " + name);
    return nullptr;
  }
  return &field_it->second;
}

bool NodePayload::fail(std::string error_message) const {
  error_message_ = std::move(error_message);
  LOG(ERROR) << error_message_;
  return false;
}

}  // namespace xllm
