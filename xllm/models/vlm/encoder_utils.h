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
#include <vector>

#include "core/framework/request/mm_batch_data.h"

namespace xllm {

class ProcessGroup;

namespace vlm {

struct EncoderShardPlan {
  std::vector<std::vector<int32_t>> rank_assignments;
  std::vector<int32_t> item_indices;
  std::vector<int32_t> input_offsets;
  std::vector<int32_t> token_nums;
  int32_t max_tokens = 0;
};

struct EncoderShardInput {
  torch::Tensor pixel_values;
  torch::Tensor grid_thw;
};

std::vector<int32_t> get_mm_token_nums(const MMBatchData& mm_data, MMType type);

EncoderShardPlan build_encoder_shard_plan(
    const std::vector<int32_t>& token_nums,
    int32_t merge_size,
    ProcessGroup* dp_group);

EncoderShardInput shard_encoder_input(const torch::Tensor& pixel_values,
                                      const torch::Tensor& grid_thw,
                                      const EncoderShardPlan& plan);

std::vector<torch::Tensor> split_by_token_nums(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& token_nums);

std::vector<torch::Tensor> gather_encoder_outputs(
    const torch::Tensor& local_tensor,
    const std::vector<int32_t>& global_token_nums,
    const EncoderShardPlan& plan,
    ProcessGroup* dp_group,
    const torch::TensorOptions& output_options,
    int64_t feature_dim);

}  // namespace vlm
}  // namespace xllm
