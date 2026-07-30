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

#include "framework/block/hierarchy_block_manager_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "distributed_runtime/engine.h"
#include "framework/multimodal/mm_data.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"

namespace xllm {

namespace {

constexpr uint32_t kBlockSize = 4;
constexpr size_t kPromptBlockCount = 4;
// An exact-prompt prefix hit drops the final block so forward can regenerate
// the last token and continue decoding.
constexpr size_t kExpectedFullPrefixBlocks = kPromptBlockCount - 1;
constexpr size_t kBlocksBeforeImage = 1;

class RecordingEngine final : public Engine {
 public:
  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}

  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }

  std::vector<folly::SemiFuture<uint32_t>> transfer_kv_blocks(
      const uint32_t /*dp_rank*/,
      const std::vector<BlockTransferInfo>& block_transfer_info) override {
    const uint32_t copied_blocks =
        static_cast<uint32_t>(block_transfer_info.size());
    copied_blocks_.store(copied_blocks, std::memory_order_release);

    std::vector<folly::SemiFuture<uint32_t>> futures;
    futures.reserve(1);
    futures.emplace_back(folly::makeSemiFuture(copied_blocks));
    return futures;
  }

  uint32_t copied_blocks() const {
    return copied_blocks_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<uint32_t> copied_blocks_{0};
};

MMData make_image_data(uint8_t image_hash_tag) {
  MMItemVec items;
  items.emplace_back(MMType::IMAGE);

  MMItemState& state = items.back().mutable_state();
  MMItemState::TokenPos& token_pos = state.mutable_token_pos();
  token_pos.offset = static_cast<int32_t>(kBlockSize);
  token_pos.length = static_cast<int32_t>(kBlockSize);
  std::fill(std::begin(state.mutable_schedule_data().key.data),
            std::end(state.mutable_schedule_data().key.data),
            image_hash_tag);

  return MMData(static_cast<uint32_t>(MMType::IMAGE), items);
}

BlockManagerPool::Options make_pool_options() {
  BlockManagerPool::Options options;
  options.num_blocks(16)
      .host_num_blocks(16)
      .block_size(static_cast<int32_t>(kBlockSize))
      .enable_prefix_cache(true)
      .enable_host_offload(true)
      .hasher_type(BlockHasherType::MM);
  return options;
}

class HierarchyBlockManagerPoolTest : public testing::Test {
 protected:
  Sequence make_sequence(size_t index, const MMData& mm_data) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens_.size() + 8;
    params.request_id = "hierarchy_block_manager_pool_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"prompt",
        /*num_prompt_tokens=*/prompt_tokens_.size(),
        /*echo=*/false,
        /*skip_special_tokens=*/true);
    return Sequence(index,
                    prompt_tokens_,
                    /*input_embedding=*/torch::Tensor(),
                    mm_data,
                    decoder,
                    params);
  }

  bool wait_for_host_prefix(HierarchyBlockManagerPool* pool,
                            Sequence* sequence,
                            size_t expected_blocks) {
    constexpr int32_t kMaxAttempts = 2000;
    for (int32_t attempt = 0; attempt < kMaxAttempts; ++attempt) {
      pool->allocate_shared(sequence);
      if (sequence->host_kv_state().shared_blocks_num(BlockType::KV) ==
          expected_blocks) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }

  const std::vector<int32_t> prompt_tokens_ =
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

TEST_F(HierarchyBlockManagerPoolTest,
       HostPrefixMatchReusesSameImageAndRejectsDifferentInputs) {
  RecordingEngine engine;
  HierarchyBlockManagerPool pool(make_pool_options(), &engine, /*dp_size=*/1);

  const MMData image_a = make_image_data(/*image_hash_tag=*/0x1a);
  Sequence cached_sequence = make_sequence(/*index=*/0, image_a);
  cached_sequence.update_block_hashes(kBlockSize, BlockHasherType::MM);
  ASSERT_EQ(cached_sequence.block_hashes().size(), kPromptBlockCount);
  ASSERT_TRUE(pool.allocate(&cached_sequence, cached_sequence.num_tokens()));
  cached_sequence.kv_state().set_kv_cache_tokens_num(
      cached_sequence.num_tokens());
  pool.deallocate(&cached_sequence);
  pool.transfer_blocks();

  Sequence same_image_sequence = make_sequence(/*index=*/1, image_a);
  same_image_sequence.update_block_hashes(kBlockSize, BlockHasherType::MM);
  ASSERT_EQ(same_image_sequence.block_hashes().size(), kPromptBlockCount);
  ASSERT_TRUE(wait_for_host_prefix(
      &pool, &same_image_sequence, kExpectedFullPrefixBlocks));
  EXPECT_EQ(engine.copied_blocks(), kPromptBlockCount);
  EXPECT_EQ(same_image_sequence.host_kv_state().kv_cache_tokens_num(),
            kExpectedFullPrefixBlocks * kBlockSize);

  const MMData image_b = make_image_data(/*image_hash_tag=*/0x2b);
  Sequence different_image_sequence = make_sequence(/*index=*/2, image_b);
  different_image_sequence.update_block_hashes(kBlockSize, BlockHasherType::MM);
  ASSERT_EQ(different_image_sequence.block_hashes().size(), kPromptBlockCount);
  EXPECT_FALSE(different_image_sequence.block_hashes()[kBlocksBeforeImage] ==
               same_image_sequence.block_hashes()[kBlocksBeforeImage]);
  pool.allocate_shared(&different_image_sequence);
  EXPECT_EQ(
      different_image_sequence.host_kv_state().shared_blocks_num(BlockType::KV),
      kBlocksBeforeImage);

  Sequence text_only_sequence = make_sequence(/*index=*/3, MMData());
  text_only_sequence.update_block_hashes(kBlockSize, BlockHasherType::MM);
  ASSERT_EQ(text_only_sequence.block_hashes().size(), kPromptBlockCount);
  EXPECT_FALSE(text_only_sequence.block_hashes()[kBlocksBeforeImage] ==
               same_image_sequence.block_hashes()[kBlocksBeforeImage]);
  pool.allocate_shared(&text_only_sequence);
  EXPECT_EQ(text_only_sequence.host_kv_state().shared_blocks_num(BlockType::KV),
            kBlocksBeforeImage);
}

}  // namespace

}  // namespace xllm
