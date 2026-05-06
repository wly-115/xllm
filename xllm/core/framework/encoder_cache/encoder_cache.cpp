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

#include "framework/encoder_cache/encoder_cache.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {
namespace {

int64_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

}  // namespace

EncoderCache::EncoderCache(int64_t max_bytes) : max_bytes_(max_bytes) {}

std::optional<torch::Tensor> EncoderCache::lookup(const XXH3Key& key) {
  EntryMap::iterator it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }

  touch(it);
  return it->second.tensor;
}

void EncoderCache::insert(const XXH3Key& key, torch::Tensor embedding) {
  EntryMap::iterator it = entries_.find(key);
  if (it != entries_.end()) {
    return;
  }

  torch::Tensor owned = embedding.clone();
  const int64_t bytes = tensor_bytes(owned);
  if (max_bytes_ > 0 && bytes > max_bytes_) {
    return;
  }

  evict_until_fit(bytes);

  lru_keys_.push_back(key);
  entries_.emplace(key,
                   Entry{std::move(owned), bytes, std::prev(lru_keys_.end())});
  current_bytes_ += bytes;
}

void EncoderCache::clear() {
  entries_.clear();
  lru_keys_.clear();
  current_bytes_ = 0;
}

void EncoderCache::touch(EntryMap::iterator it) {
  lru_keys_.splice(lru_keys_.end(), lru_keys_, it->second.lru_it);
}

void EncoderCache::erase(EntryMap::iterator it) {
  current_bytes_ -= it->second.bytes;
  lru_keys_.erase(it->second.lru_it);
  entries_.erase(it);
}

void EncoderCache::evict_until_fit(int64_t incoming_bytes) {
  if (max_bytes_ <= 0) {
    return;
  }
  while (!lru_keys_.empty() && current_bytes_ + incoming_bytes > max_bytes_) {
    const XXH3Key& evict_key = lru_keys_.front();
    EntryMap::iterator it = entries_.find(evict_key);
    CHECK(it != entries_.end());
    erase(it);
  }
}

}  // namespace xllm
