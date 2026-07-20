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

#include <gtest/gtest.h>

#include "core/framework/multimodal/mm_data.h"

namespace xllm {
namespace {

TEST(NodePayloadTest, PreservesMultimodalTokenCount) {
  MMDataItem item(MMType::IMAGE);
  item.add("pixel_values", torch::ones({1, 1}));
  item.mutable_state().mutable_token_pos() = {4, 4};
  item.mutable_state().mutable_mm_token_num() = 4;
  item.mutable_state().mutable_seq_index() = 2;
  item.mutable_state().mutable_mm_token_mask() =
      torch::ones({4}, torch::TensorOptions().dtype(torch::kBool));

  MMData data(MMType::IMAGE, MMItemVec{item});
  NodePayload payload;
  ASSERT_TRUE(payload.set("mm_data", data));

  proto::NodePayload proto_payload;
  ASSERT_TRUE(payload.to_proto(&proto_payload));

  std::optional<NodePayload> decoded_payload =
      NodePayload::from_proto(proto_payload);
  ASSERT_TRUE(decoded_payload.has_value());
  std::optional<MMData> decoded_data = decoded_payload->get<MMData>("mm_data");
  ASSERT_TRUE(decoded_data.has_value());

  ASSERT_TRUE(decoded_data->hold<MMItemVec>());
  ASSERT_EQ(decoded_data->items<MMItemVec>().size(), 1);
  const MMDataItem& decoded_item = decoded_data->items<MMItemVec>().front();
  EXPECT_EQ(decoded_item.state().mm_token_num(), 4);
  EXPECT_EQ(decoded_item.state().seq_index(), 2);
  EXPECT_EQ(decoded_item.state().token_pos().offset, 4);
  EXPECT_EQ(decoded_item.state().token_pos().length, 4);
  EXPECT_TRUE(torch::equal(
      decoded_item.state().mm_token_mask(),
      torch::ones({4}, torch::TensorOptions().dtype(torch::kBool))));
}

}  // namespace
}  // namespace xllm
