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

#include "core/framework/ensemble/graph_config.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace xllm {
namespace {

constexpr char kQwenImageEditConfig[] = R"yaml(
graph_name: qwen_image_edit
ready_timeout_ms: 60000
preprocessor_config:
  type: qwen-image-edit
  model_root: /path/to/qwen_image_edit
result_endpoint:
  target: result
nodes:
  - name: node0
    adapter: qwen_vlm_encode
    next: [node1]
    ranks: [0]
    engine_config:
      backend: vlm
      model: /path/to/qwen_image_edit_vlm
      model_id: qwen_image_edit_vlm
      task: mm_embed
      limit_image_per_prompt: 1
      enable_return_mm_full_embeddings: true
    endpoint:
      target: node0
  - name: node1
    adapter: qwen_image_edit_dit
    ranks: [1]
    engine_config:
      backend: dit
      model: /path/to/qwen_image_edit_dit
      model_id: qwen_image_edit_dit
      task: generate
      max_requests_per_batch: 1
      dit_cache_policy: None
    endpoint:
      target: node1
    final_output: true
    timeout_ms: 120000
)yaml";

GraphConfig load_config_or_die(const std::string& yaml_text) {
  GraphConfig config;
  load_graph_config_from_yaml(yaml_text, config);
  return config;
}

void expect_load_config_fatal(const std::string& yaml_text) {
  EXPECT_DEATH(
      {
        GraphConfig config;
        load_graph_config_from_yaml(yaml_text, config);
      },
      "");
}

void expect_validate_config_from_yaml_fatal(const std::string& yaml_text) {
  EXPECT_DEATH(
      {
        GraphConfig config;
        load_graph_config_from_yaml(yaml_text, config);
        validate_graph_config(config);
      },
      "");
}

void expect_validate_config_fatal(const GraphConfig& config) {
  EXPECT_DEATH(
      {
        GraphConfig copied_config = config;
        validate_graph_config(copied_config);
      },
      "");
}

TEST(GraphConfigLoadTest, LoadQwenImageEditConfig) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  EXPECT_EQ(config.graph_name, "qwen_image_edit");
  EXPECT_EQ(config.result_target, "result");
  ASSERT_EQ(config.nodes.size(), 2);
  validate_graph_config(config);

  const NodeConfig& vlm_node = config.nodes[0];
  EXPECT_EQ(vlm_node.name, "node0");
  EXPECT_EQ(vlm_node.backend, "vlm");
  EXPECT_EQ(vlm_node.adapter, "qwen_vlm_encode");
  ASSERT_EQ(vlm_node.next_nodes.size(), 1);
  EXPECT_EQ(vlm_node.next_nodes[0], "node1");
  ASSERT_EQ(vlm_node.ranks.size(), 1);
  EXPECT_EQ(vlm_node.ranks.at(0), 0);
  EXPECT_EQ(vlm_node.engine_config.at("model"), "/path/to/qwen_image_edit_vlm");
  EXPECT_EQ(vlm_node.engine_config.at("enable_return_mm_full_embeddings"),
            "true");
  EXPECT_EQ(vlm_node.endpoint_target, "node0");
  EXPECT_FALSE(vlm_node.final_output);
  EXPECT_EQ(vlm_node.timeout_ms, 30000);

  const NodeConfig& dit_node = config.nodes[1];
  EXPECT_EQ(dit_node.backend, "dit");
  EXPECT_EQ(dit_node.adapter, "qwen_image_edit_dit");
  EXPECT_TRUE(dit_node.next_nodes.empty());
  ASSERT_EQ(dit_node.ranks.size(), 1);
  EXPECT_EQ(dit_node.ranks.at(1), 0);
  EXPECT_EQ(dit_node.endpoint_target, "node1");
  EXPECT_EQ(dit_node.engine_config.at("max_requests_per_batch"), "1");
  EXPECT_TRUE(dit_node.final_output);
  EXPECT_EQ(dit_node.timeout_ms, 120000);
}

TEST(GraphConfigIndexTest, BuildsRuntimeIndices) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);

  EXPECT_EQ(config.node_index.at("node0"), 0);
  EXPECT_EQ(config.node_index.at("node1"), 1);
  ASSERT_EQ(config.downstream_nodes.size(), 2);
  ASSERT_EQ(config.get_downstream("node0").size(), 1);
  EXPECT_EQ(config.get_downstream("node0")[0], "node1");
  EXPECT_TRUE(config.get_downstream("node1").empty());
  ASSERT_EQ(config.get_root_nodes().size(), 1);
  EXPECT_EQ(config.get_root_nodes()[0], "node0");
  ASSERT_EQ(config.get_final_nodes().size(), 1);
  EXPECT_EQ(config.get_final_nodes()[0], "node1");
  ASSERT_NE(config.find_node("node1"), nullptr);
  EXPECT_EQ(config.find_node("missing"), nullptr);
}

TEST(GraphConfigLoadTest, RejectsDuplicateRankInNode) {
  constexpr char kConfig[] = R"yaml(
graph_name: duplicate_rank
nodes:
  - name: node0
    ranks: [0, 0]
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsDuplicateNodeName) {
  constexpr char kConfig[] = R"yaml(
graph_name: duplicate_node
nodes:
  - name: node0
    ranks: [0]
    endpoint:
      target: node0
  - name: node0
    ranks: [1]
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_validate_config_from_yaml_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsRankBelongingToMultipleNodes) {
  GraphConfig config;
  config.graph_name = "duplicate_global_rank";

  NodeConfig node0;
  node0.name = "node0";
  node0.ranks = {{0, 0}};
  config.nodes.emplace_back(node0);

  NodeConfig node1;
  node1.name = "node1";
  node1.ranks = {{0, 0}};
  node1.final_output = true;
  config.nodes.emplace_back(node1);

  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsNodeWithEmptyRanks) {
  constexpr char kConfig[] = R"yaml(
graph_name: empty_ranks
nodes:
  - name: node0
    ranks: []
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_validate_config_from_yaml_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsUnknownNextNode) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  config.nodes[0].next_nodes = {"missing"};
  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsMultipleDownstreamNodes) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  config.nodes[0].next_nodes = {"node1", "node0"};
  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsMultipleUpstreamNodes) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  NodeConfig second_root = config.nodes[0];
  second_root.name = "node2";
  second_root.ranks = {{2, 0}};
  second_root.endpoint_target = "node2";
  config.nodes.emplace_back(std::move(second_root));
  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsMissingFinalOutput) {
  constexpr char kConfig[] = R"yaml(
graph_name: missing_output
nodes:
  - name: node0
    ranks: [0]
    endpoint:
      target: node0
)yaml";
  expect_validate_config_from_yaml_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsFinalOutputWithDownstreamNode) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  config.nodes[0].final_output = true;
  config.nodes[1].final_output = false;
  expect_validate_config_fatal(config);
}

TEST(GraphConfigLoadTest, RejectsRankOutsideInt32Range) {
  constexpr char kConfig[] = R"yaml(
graph_name: rank_overflow
nodes:
  - name: node0
    ranks: [2147483648]
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeRank) {
  constexpr char kConfig[] = R"yaml(
graph_name: negative_rank
nodes:
  - name: node0
    ranks: [-1]
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeReadyTimeout) {
  constexpr char kConfig[] = R"yaml(
graph_name: negative_ready_timeout
ready_timeout_ms: -1
nodes:
  - name: node0
    ranks: [0]
    endpoint:
      target: node0
    final_output: true
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeNodeTimeout) {
  constexpr char kConfig[] = R"yaml(
graph_name: negative_node_timeout
nodes:
  - name: node0
    ranks: [0]
    endpoint:
      target: node0
    final_output: true
    timeout_ms: -1
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsCyclicGraph) {
  GraphConfig config = load_config_or_die(kQwenImageEditConfig);
  config.nodes[1].next_nodes = {"node0"};
  expect_validate_config_fatal(config);
}

TEST(GraphConfigLoadTest, RejectsLegacyDepsField) {
  constexpr char kConfig[] = R"yaml(
graph_name: legacy_deps
nodes:
  - name: node0
    adapter: qwen_vlm_encode
    deps: [node1]
    ranks: [0]
    engine_config:
      backend: vlm
    endpoint:
      target: node0
)yaml";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, BuildsLocalRankLookupFromSortedGlobalRanks) {
  constexpr char kConfig[] = R"yaml(
graph_name: rank_lookup
preprocessor_config:
  type: qwen-image-edit
  model_root: /path/to/qwen_image_edit
result_endpoint:
  target: result
nodes:
  - name: node0
    adapter: qwen_vlm_encode
    ranks: [4, 2, 3]
    engine_config:
      backend: vlm
    endpoint:
      target: node0
    final_output: true
)yaml";
  GraphConfig config = load_config_or_die(kConfig);

  ASSERT_EQ(config.nodes.size(), 1);
  const NodeConfig& node = config.nodes[0];
  ASSERT_EQ(node.ranks.size(), 3);
  EXPECT_EQ(node.ranks.at(2), 0);
  EXPECT_EQ(node.ranks.at(3), 1);
  EXPECT_EQ(node.ranks.at(4), 2);
}

}  // namespace
}  // namespace xllm
