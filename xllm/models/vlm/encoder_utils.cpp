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

#include "models/vlm/encoder_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <numeric>

#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace vlm {

std::vector<int32_t> get_mm_token_nums(const MMBatchData& mm_data,
                                       MMType type) {
  std::vector<int32_t> token_nums;

  const std::vector<MMData>& mm_data_vec = mm_data.mm_data_vec();
  for (const MMData& mm_data : mm_data_vec) {
    if (!mm_data.hold<MMItemVec>()) {
      return {};
    }

    const MMItemVec& items = mm_data.items<MMItemVec>();
    for (const MMDataItem& item : items) {
      if (item.type() != type) {
        continue;
      }
      const int32_t mm_token_num = item.state().mm_token_num();
      CHECK_GT(mm_token_num, 0) << "missing multimodal token num";
      token_nums.push_back(mm_token_num);
    }
  }
  return token_nums;
}

EncoderShardPlan build_encoder_shard_plan(
    const std::vector<int32_t>& token_nums,
    int32_t merge_size,
    ProcessGroup* dp_group) {
  EncoderShardPlan plan;
  const int32_t item_count = static_cast<int32_t>(token_nums.size());
  const int32_t world_size = dp_group->world_size();
  plan.rank_assignments.resize(world_size);
  plan.input_offsets.resize(item_count + 1, 0);
  const int32_t merge_unit = merge_size * merge_size;
  for (int32_t item_index = 0; item_index < item_count; ++item_index) {
    plan.input_offsets[item_index + 1] =
        plan.input_offsets[item_index] + token_nums[item_index] * merge_unit;
  }

  std::vector<int32_t> item_order(item_count);
  std::iota(item_order.begin(), item_order.end(), 0);
  std::sort(
      item_order.begin(), item_order.end(), [&](int32_t lhs, int32_t rhs) {
        if (token_nums[lhs] != token_nums[rhs]) {
          return token_nums[lhs] > token_nums[rhs];
        }
        return lhs < rhs;
      });

  std::vector<int32_t> rank_loads(world_size, 0);
  for (int32_t item_index : item_order) {
    const auto target_it =
        std::min_element(rank_loads.begin(), rank_loads.end());
    const int32_t target_rank =
        static_cast<int32_t>(std::distance(rank_loads.begin(), target_it));
    plan.rank_assignments[target_rank].push_back(item_index);
    rank_loads[target_rank] += token_nums[item_index];
    plan.max_tokens = std::max(plan.max_tokens, rank_loads[target_rank]);
  }

  for (std::vector<int32_t>& item_indices : plan.rank_assignments) {
    std::sort(item_indices.begin(), item_indices.end());
  }

  plan.item_indices = plan.rank_assignments[dp_group->rank()];
  plan.token_nums.reserve(plan.item_indices.size());
  for (int32_t item_index : plan.item_indices) {
    plan.token_nums.push_back(token_nums[item_index]);
  }
  return plan;
}

EncoderShardInput shard_encoder_input(const torch::Tensor& pixel_values,
                                      const torch::Tensor& grid_thw,
                                      const EncoderShardPlan& plan) {
  EncoderShardInput input;
  input.pixel_values = pixel_values;
  input.grid_thw = grid_thw;

  if (plan.item_indices.empty()) {
    input.pixel_values = pixel_values.slice(0, 0, 0);
    input.grid_thw = grid_thw.slice(0, 0, 0);
    return input;
  }

  std::vector<torch::Tensor> pixel_slices;
  std::vector<torch::Tensor> grid_slices;
  pixel_slices.reserve(plan.item_indices.size());
  grid_slices.reserve(plan.item_indices.size());
  for (int32_t item_index : plan.item_indices) {
    pixel_slices.push_back(pixel_values.slice(
        0, plan.input_offsets[item_index], plan.input_offsets[item_index + 1]));
    grid_slices.push_back(grid_thw.slice(0, item_index, item_index + 1));
  }

  if (pixel_slices.size() == 1) {
    input.pixel_values = pixel_slices.front();
    input.grid_thw = grid_slices.front();
  } else {
    input.pixel_values = torch::cat(pixel_slices, 0);
    input.grid_thw = torch::cat(grid_slices, 0);
  }
  return input;
}

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums) {
  std::vector<int64_t> split_sizes(token_nums.begin(), token_nums.end());
  return tensor.split(split_sizes, 0);
}

namespace {

std::vector<torch::Tensor> restore_encoder_outputs(
    const torch::Tensor& gathered_payload,
    const std::vector<int32_t>& global_token_nums,
    const std::vector<std::vector<int32_t>>& rank_assignments,
    int32_t max_tokens) {
  std::vector<torch::Tensor> outputs(global_token_nums.size());

  const int32_t world_size = static_cast<int32_t>(rank_assignments.size());
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const int32_t payload_base = rank * max_tokens;
    const std::vector<int32_t>& rank_indices = rank_assignments[rank];
    int32_t token_offset = 0;

    for (int32_t item_index : rank_indices) {
      const int32_t token_num = global_token_nums[item_index];
      outputs[item_index] =
          gathered_payload.slice(0,
                                 payload_base + token_offset,
                                 payload_base + token_offset + token_num);
      token_offset += token_num;
    }
  }
  return outputs;
}

}  // namespace

std::vector<torch::Tensor> gather_encoder_outputs(
    const torch::Tensor& local_tensor,
    const std::vector<int32_t>& global_token_nums,
    const EncoderShardPlan& plan,
    ProcessGroup* dp_group,
    const torch::TensorOptions& output_options,
    int64_t feature_dim) {
  const int32_t local_total_tokens = std::accumulate(
      plan.token_nums.begin(), plan.token_nums.end(), int32_t{0});
  torch::Tensor local_payload =
      torch::zeros({plan.max_tokens, feature_dim}, output_options);
  if (local_total_tokens > 0) {
    local_payload.slice(0, 0, local_total_tokens).copy_(local_tensor);
  }

  torch::Tensor gathered_payload =
      parallel_state::gather(local_payload, dp_group, 0);
  return restore_encoder_outputs(gathered_payload,
                                 global_token_nums,
                                 plan.rank_assignments,
                                 plan.max_tokens);
}

}  // namespace vlm
}  // namespace xllm
