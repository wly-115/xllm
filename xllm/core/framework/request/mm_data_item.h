/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mm_type.h"
#include "util/hash_util.h"

namespace xllm {

using MMKey = std::string;
using MMValue = std::variant<torch::Tensor, std::vector<torch::Tensor>>;
using MMDict = std::unordered_map<MMKey, MMValue>;

struct MMPosition {
  int32_t offset;
  int32_t length;
};

class MMDataItem {
  using MMMetadata = std::variant<ImageMetadata, VideoMetadata, AudioMetadata>;

 public:
  class IVisitor {
   public:
    virtual ~IVisitor() = default;
    virtual bool visit(MMDataItem& item) = 0;
  };

 public:
  MMDataItem() = default;
  MMDataItem(MMType ty);
  MMDataItem(MMType ty, const MMDict& data);
  MMDataItem(MMType ty, const MMDict& data, const MMMetadata& metadata);

  bool valid() const { return ty_ != MMType::NONE; }
  bool is_type(MMType type) const { return ty_ == type; }

  const MMDict& data() const { return data_; }
  void set_data(const MMDict& data) { data_ = data; }

  MMType type() const { return ty_; }
  bool has(const MMKey& key) const;

  template <typename T>
  std::optional<T> get(const MMKey& key) const {
    if (!valid()) return std::nullopt;

    const auto& itor = data_.find(key);
    if (itor != data_.end())
      return std::get<T>(itor->second);
    else
      return std::nullopt;
  }

  template <typename T>
  void update(const MMKey& key, const T& value) {
    const auto& itor = data_.find(key);
    if (itor != data_.end()) {
      // Key exists, update it
      data_[key] = value;
    } else {
      data_.insert({key, value});
    }
  }

  template <typename T>
  std::optional<T> get_metadata() const {
    if (!valid()) return std::nullopt;

    if (std::holds_alternative<T>(metadata_)) {
      return std::get<T>(metadata_);
    } else {
      return std::nullopt;
    }
  }
  const MMMetadata& get_metadata() const { return metadata_; }

  template <typename T>
  void set_metadata(const T& meta) {
    metadata_ = meta;
  }
  const MMPosition& get_mm_position() const { return mm_position_; }
  void set_mm_position(const MMPosition& mm_position) {
    mm_position_ = mm_position;
  }

  int32_t get_cached_tokens_num() const { return cached_tokens_num_; }

  void set_cached_tokens_num(int cached_tokens_num) {
    cached_tokens_num_ = cached_tokens_num;
  }

  const uint8_t* get_immutable_hash_value() const { return hash_value_; }
  uint8_t* get_mutable_hash_value() { return hash_value_; }

  void set_mm_hash_value(const uint8_t* hash_value) {
    memcpy(hash_value_, hash_value, MURMUR_HASH3_VALUE_LEN);
  }

  void debug_print() const;

 private:
  MMType ty_ = MMType::NONE;
  int32_t cached_tokens_num_ = 0;
  uint8_t hash_value_[MURMUR_HASH3_VALUE_LEN];
  MMPosition mm_position_;
  MMMetadata metadata_;
  MMDict data_;
};

}  // namespace xllm
