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

#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>
#include <xxHash/xxhash.h>

#include <iostream>
#include <thread>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "request/mm_data.h"
namespace xllm {
namespace {

void mm_xxh3_128bits_hash(const std::vector<const uint8_t*>& mm_hash_values,
                          const uint8_t* pre_hash_value,
                          const Slice<int32_t>& token_ids,
                          uint8_t* hash_value) {
  xxh3_128bits_hash(pre_hash_value, token_ids, hash_value);
  if (mm_hash_values.empty()) {
    return;
  }

  const size_t data_len =
      XXH3_128BITS_HASH_VALUE_LEN * (mm_hash_values.size() + 1);
  std::vector<uint8_t> key_buffer;
  key_buffer.reserve(data_len);
  key_buffer.insert(
      key_buffer.end(), hash_value, hash_value + XXH3_128BITS_HASH_VALUE_LEN);
  for (const uint8_t* mm_hash_value : mm_hash_values) {
    key_buffer.insert(key_buffer.end(),
                      mm_hash_value,
                      mm_hash_value + XXH3_128BITS_HASH_VALUE_LEN);
  }

  XXH128_hash_t mm_hash =
      XXH3_128bits_withSeed(reinterpret_cast<const void*>(key_buffer.data()),
                            key_buffer.size(),
                            FLAGS_xxh3_128bits_seed);
  std::memcpy(hash_value, &mm_hash, sizeof(mm_hash));
}

int32_t find_overlapping_mm_idx(const MMData& mm_data,
                                int32_t start_token_idx) {
  if (!mm_data.valid()) {
    return 0;
  }

  const auto& mm_items = mm_data.items<MMItemVec>();
  const int32_t num_mm_items = mm_data.size();
  int32_t next_item_idx = 0;
  while (next_item_idx < num_mm_items) {
    const auto& pos = mm_items[next_item_idx].state().token_pos();
    const int32_t item_end_token_idx = pos.offset + pos.length;
    if (start_token_idx < item_end_token_idx) {
      break;
    }
    ++next_item_idx;
  }
  return next_item_idx;
}

std::vector<const uint8_t*> get_block_mm_hash_values(const MMData& mm_data,
                                                     int32_t start_token_idx,
                                                     int32_t end_token_idx,
                                                     int32_t& next_item_idx) {
  if (!mm_data.valid()) {
    return {};
  }

  const auto& mm_items = mm_data.items<MMItemVec>();
  std::vector<const uint8_t*> mm_hash_values;
  const int32_t num_mm_items = mm_data.size();
  while (next_item_idx < num_mm_items) {
    const auto& pos = mm_items[next_item_idx].state().token_pos();
    const int32_t item_start_token_idx = pos.offset;
    const int32_t item_end_token_idx = item_start_token_idx + pos.length;
    if (end_token_idx <= item_start_token_idx) {
      break;
    }
    if (start_token_idx >= item_end_token_idx) {
      ++next_item_idx;
      continue;
    }
    const auto& schedule_data = mm_items[next_item_idx].state().schedule_data();
    mm_hash_values.push_back(schedule_data.key.data);
    if (item_end_token_idx > end_token_idx) {
      break;
    }
    ++next_item_idx;
  }
  return mm_hash_values;
}

void compute_block_hash(const MMData& mm_data,
                        const Slice<int32_t>& token_ids,
                        int32_t start_token_idx,
                        int32_t end_token_idx,
                        int32_t& next_item_idx,
                        XXH3Key* token_hash_key) {
  const uint8_t* pre_hash_value =
      start_token_idx == 0 ? nullptr : token_hash_key->data;
  const Slice<int32_t> block_token_ids =
      token_ids.slice(start_token_idx, end_token_idx);
  std::vector<const uint8_t*> mm_hash_values = get_block_mm_hash_values(
      mm_data, start_token_idx, end_token_idx, next_item_idx);
  mm_xxh3_128bits_hash(
      mm_hash_values, pre_hash_value, block_token_ids, token_hash_key->data);
}

}  // namespace

void xxh3_128bits_hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    XXH128_hash_t xxh3_128bits_hash_value =
        XXH3_128bits_withSeed(reinterpret_cast<const void*>(token_ids.data()),
                              sizeof(int32_t) * token_ids.size(),
                              FLAGS_xxh3_128bits_seed);
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
  } else {
    uint8_t key[1024];

    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + XXH3_128BITS_HASH_VALUE_LEN;
    CHECK_GT(sizeof(key), data_len) << "key size is too small";

    memcpy(key, pre_hash_value, XXH3_128BITS_HASH_VALUE_LEN);
    memcpy(key + XXH3_128BITS_HASH_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    XXH128_hash_t xxh3_128bits_hash_value = XXH3_128bits_withSeed(
        reinterpret_cast<const void*>(key), data_len, FLAGS_xxh3_128bits_seed);
    memcpy(
        hash_value, &xxh3_128bits_hash_value, sizeof(xxh3_128bits_hash_value));
  }
}

std::vector<Block> PrefixCache::match(const Slice<int32_t>& token_ids,
                                      const Slice<Block>& existed_shared_blocks,
                                      const MMData& mm_data) {
  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  if (n_tokens == 0) {
    return std::vector<Block>();
  }

  size_t n_blocks = n_tokens / block_size_;
  total_blocks_.fetch_add(n_blocks);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  blocks.insert(
      blocks.end(), existed_shared_blocks.begin(), existed_shared_blocks.end());

  DNodeList node_list;

  size_t start_index = existed_shared_blocks.size() * block_size_;
  XXH3Key token_hash_key =
      existed_shared_blocks.empty()
          ? XXH3Key{}
          : XXH3Key{existed_shared_blocks.back().get_immutable_hash_value()};
  int32_t next_item_idx = find_overlapping_mm_idx(mm_data, start_index);
  for (size_t i = start_index; i < n_tokens; i += block_size_) {
    compute_block_hash(
        mm_data, token_ids, i, i + block_size_, next_item_idx, &token_hash_key);

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      blocks.push_back(iter->second->block);
      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      break;
    }
  }

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  matched_blocks_.fetch_add(blocks.size());

  int64_t int_rate_percent = static_cast<int64_t>(
      static_cast<double>(blocks.size()) * 100.0 / n_blocks);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_rate, int_rate_percent);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_num, blocks.size());

  return blocks;
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           size_t existed_shared_blocks_num,
                           const MMData& mm_data) {
  std::vector<XXH3Key> insert_keys;
  return insert(
      token_ids, blocks, existed_shared_blocks_num, mm_data, &insert_keys);
}

size_t PrefixCache::insert(const std::vector<Block>& blocks) {
  Slice<Block> slice(blocks);
  return insert(slice);
}

size_t PrefixCache::insert(Slice<Block>& blocks) {
  std::vector<XXH3Key> insert_keys;
  return insert(blocks, &insert_keys);
}

size_t PrefixCache::evict(size_t n_blocks) {
  std::vector<XXH3Key> evict_keys;
  return evict(n_blocks, &evict_keys);
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           size_t existed_shared_blocks_num,
                           const MMData& mm_data,
                           std::vector<XXH3Key>* insert_keys) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());
  const size_t n_tokens = n_blocks * block_size_;

  if (n_blocks == 0) {
    return 0;
  }
  CHECK_GE(n_blocks, existed_shared_blocks_num);
  // truncate the token ids and blocks to boundary

  DNodeList node_list;
  XXH3Key token_hash_key = existed_shared_blocks_num == 0
                               ? XXH3Key{}
                               : XXH3Key{blocks[existed_shared_blocks_num - 1]
                                             .get_immutable_hash_value()};
  int32_t next_item_idx =
      find_overlapping_mm_idx(mm_data, existed_shared_blocks_num * block_size_);

  uint32_t block_idx = existed_shared_blocks_num;
  insert_keys->reserve(n_blocks);
  for (size_t i = existed_shared_blocks_num * block_size_; i < n_tokens;
       i += block_size_) {
    compute_block_hash(
        mm_data, token_ids, i, i + block_size_, next_item_idx, &token_hash_key);
    blocks[block_idx].set_hash_value(token_hash_key.data);

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;

      insert_keys->emplace_back(token_hash_key.data);
    }

    ++block_idx;
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return n_tokens;
}

size_t PrefixCache::insert(Slice<Block>& blocks,
                           std::vector<XXH3Key>* insert_keys) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  DNodeList node_list;
  XXH3Key token_hash_key;

  insert_keys->reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); i++) {
    if (!blocks[i].is_valid()) {
      continue;
    }
    token_hash_key.set(blocks[i].get_immutable_hash_value());

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[i];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;

      insert_keys->emplace_back(token_hash_key.data);
    }
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return blocks.size() * block_size_;
}

size_t PrefixCache::evict(size_t n_blocks, std::vector<XXH3Key>* evict_keys) {
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  evict_keys->reserve(n_blocks);
  while (evict_count < n_blocks) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    XXH3Key token_hash_key(del_node->block.get_immutable_hash_value());

    cached_blocks_.erase(token_hash_key);

    delete del_node;
    ++evict_count;
    --num_blocks_;

    evict_keys->emplace_back(std::move(token_hash_key));
  }

  return evict_count;
}

uint32_t PrefixCache::compute_hash_keys(const Slice<int32_t>& token_ids,
                                        std::vector<Block>& blocks,
                                        const size_t cached_blocks) {
  if (blocks.size() == 0) {
    return 0;
  }
  int32_t block_size = blocks[0].size();
  const size_t n_blocks = (token_ids.size() + block_size - 1) / block_size;
  if (blocks.size() > n_blocks) {
    LOG(ERROR) << "token ids do not cover the allocate block.";
    return 0;
  }
  size_t full_block_size =
      std::min(token_ids.size() / block_size, blocks.size());

  for (size_t i = cached_blocks; i < full_block_size; i++) {
    if (i == 0) {
      xxh3_128bits_hash(nullptr,
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    } else {
      xxh3_128bits_hash(blocks[i - 1].get_mutable_hash_value(),
                        token_ids.slice(i * block_size, (i + 1) * block_size),
                        blocks[i].get_mutable_hash_value());
    }
  }

  return full_block_size;
}

}  // namespace xllm
