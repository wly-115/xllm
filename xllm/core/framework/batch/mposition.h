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

#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace xllm {

class Sequence;
enum class MPositionType : int8_t;
struct ModelArgs;

class MPositionGenerator {
 public:
  virtual ~MPositionGenerator() = default;

  virtual std::tuple<torch::Tensor, int32_t> generate(
      Sequence& sequence,
      const ModelArgs& model_args) const = 0;
};

std::unique_ptr<MPositionGenerator> create_mposition_generator(
    MPositionType mposition_type);

class MPositionHelper {
 public:
  MPositionHelper(Sequence& seq,
                  const ModelArgs& args,
                  std::unique_ptr<MPositionGenerator> generator)
      : seq_(seq), args_(args), generator_(std::move(generator)) {}

  torch::Tensor get_positions();

 private:
  torch::Tensor get_positions_d();

 private:
  Sequence& seq_;
  const ModelArgs& args_;
  std::unique_ptr<MPositionGenerator> generator_;
};

}  // namespace xllm
