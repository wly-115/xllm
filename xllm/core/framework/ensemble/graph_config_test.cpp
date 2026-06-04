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

namespace xllm {
namespace ensemble {
namespace {

constexpr char kQwenImageEditConfig[] = R"json(
{
  "graph_name": "qwen_image_edit",
  "ready_timeout_ms": 60000,
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "engine_config": {
        "backend": "vlm",
        "model": "/path/to/qwen_image_edit_vlm",
        "model_id": "qwen_image_edit_vlm",
        "task": "mm_embed",
        "limit_image_per_prompt": "1",
        "enable_return_mm_full_embeddings": true
      },
      "request_adapter": "qwen_image_edit_vlm_request",
      "result_adapter": "qwen_image_edit_vlm_result",
      "result_adapter_config": {
        "prompt_embeds_key": "node0.prompt_embeds"
      },
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      }
    },
    {
      "name": "node1",
      "node_type": "dit",
      "deps": ["node0"],
      "ranks": [1],
      "engine_config": {
        "backend": "dit",
        "model": "/path/to/qwen_image_edit_dit",
        "model_id": "qwen_image_edit_dit",
        "task": "generate",
        "max_requests_per_batch": 1,
        "dit_cache_policy": "None"
      },
      "request_adapter": "qwen_image_edit_dit_request",
      "request_adapter_config": {
        "prompt_embeds_source": "node0.prompt_embeds"
      },
      "result_adapter": "qwen_image_edit_dit_result",
      "result_adapter_config": {
        "images_key": "node1.images"
      },
      "endpoint": {
        "transport": "brpc",
        "target": "node1",
        "args": {
          "service_name": "qwen_image_edit_dit"
        }
      },
      "final_output": true,
      "output_keys": ["images"],
      "timeout_ms": 120000
    }
  ]
}
)json";

GraphConfig load_config_or_die(const std::string& json_text) {
  GraphConfig config;
  load_graph_config_from_json(json_text, config);
  return config;
}

void expect_load_config_fatal(const std::string& json_text) {
  EXPECT_DEATH(
      {
        GraphConfig config;
        load_graph_config_from_json(json_text, config);
      },
      "");
}

void expect_validate_config_from_json_fatal(const std::string& json_text) {
  EXPECT_DEATH(
      {
        GraphConfig config;
        load_graph_config_from_json(json_text, config);
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
  ASSERT_EQ(config.nodes.size(), 2);
  validate_graph_config(config);

  const NodeConfig& vlm_node = config.nodes[0];
  EXPECT_EQ(vlm_node.name, "node0");
  EXPECT_TRUE(vlm_node.deps.empty());
  ASSERT_EQ(vlm_node.ranks.size(), 1);
  EXPECT_EQ(vlm_node.ranks.at(0), 0);
  EXPECT_EQ(vlm_node.engine_config.at("model"), "/path/to/qwen_image_edit_vlm");
  EXPECT_EQ(vlm_node.engine_config.at("enable_return_mm_full_embeddings"),
            "true");
  EXPECT_FALSE(vlm_node.final_output);
  EXPECT_EQ(vlm_node.timeout_ms, 30000);

  const NodeConfig& dit_node = config.nodes[1];
  ASSERT_EQ(dit_node.deps.size(), 1);
  EXPECT_EQ(dit_node.deps[0], "node0");
  ASSERT_EQ(dit_node.ranks.size(), 1);
  EXPECT_EQ(dit_node.ranks.at(1), 0);
  EXPECT_EQ(dit_node.endpoint.args.at("service_name"), "qwen_image_edit_dit");
  EXPECT_EQ(dit_node.engine_config.at("max_requests_per_batch"), "1");
  EXPECT_TRUE(dit_node.final_output);
  ASSERT_EQ(dit_node.output_keys.size(), 1);
  EXPECT_EQ(dit_node.output_keys[0], "images");
  EXPECT_EQ(dit_node.timeout_ms, 120000);
}

TEST(GraphConfigLoadTest, RejectsDuplicateRankInNode) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "duplicate_rank",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0, 0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsDuplicateNodeName) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "duplicate_node",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      }
    },
    {
      "name": "node0",
      "node_type": "dit",
      "ranks": [1],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "brpc",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["images"]
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
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
  node1.output_keys = {"text"};
  config.nodes.emplace_back(node1);

  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsInvalidLocalRankMapping) {
  GraphConfig config;
  config.graph_name = "invalid_local_rank";

  NodeConfig node;
  node.name = "node0";
  node.ranks = {{2, 2}};
  node.final_output = true;
  node.output_keys = {"text"};
  config.nodes.emplace_back(node);

  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsNegativeGlobalRankMapping) {
  GraphConfig config;
  config.graph_name = "negative_global_rank";

  NodeConfig node;
  node.name = "node0";
  node.ranks = {{-1, 0}};
  node.final_output = true;
  node.output_keys = {"text"};
  config.nodes.emplace_back(node);

  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsDuplicateLocalRankMapping) {
  GraphConfig config;
  config.graph_name = "duplicate_local_rank";

  NodeConfig node;
  node.name = "node0";
  node.ranks = {{2, 0}, {3, 0}};
  node.final_output = true;
  node.output_keys = {"text"};
  config.nodes.emplace_back(node);

  expect_validate_config_fatal(config);
}

TEST(GraphConfigValidateTest, RejectsNodeWithEmptyRanks) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "empty_ranks",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsUnknownDependency) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "unknown_dep",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "deps": ["missing"],
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsMissingFinalOutput) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "missing_output",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      }
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsEmptyFinalOutputKeys) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "empty_output_keys",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": []
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsFinalOutputWithDownstreamNode) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "bad_output_topology",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    },
    {
      "name": "node1",
      "node_type": "dit",
      "deps": ["node0"],
      "ranks": [1],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "brpc",
        "target": "node1"
      }
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsRankOutsideInt32Range) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "rank_overflow",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [2147483648],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeRank) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "negative_rank",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [-1],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeReadyTimeout) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "negative_ready_timeout",
  "ready_timeout_ms": -1,
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigLoadTest, RejectsNegativeNodeTimeout) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "negative_node_timeout",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"],
      "timeout_ms": -1
    }
  ]
}
)json";
  expect_load_config_fatal(kConfig);
}

TEST(GraphConfigValidateTest, RejectsCyclicGraph) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "cycle_graph",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "deps": ["node1"],
      "ranks": [0],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      }
    },
    {
      "name": "node1",
      "node_type": "dit",
      "deps": ["node0"],
      "ranks": [1],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "brpc",
        "target": "node1"
      }
    },
    {
      "name": "node2",
      "node_type": "dit",
      "ranks": [2],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "brpc",
        "target": "node2"
      },
      "final_output": true,
      "output_keys": ["images"]
    }
  ]
}
)json";
  expect_validate_config_from_json_fatal(kConfig);
}

TEST(GraphConfigTest, BuildsReadyKey) {
  EXPECT_EQ(make_ready_key("qwen_image_edit", "node1"),
            "qwen_image_edit/node1");
}

TEST(GraphConfigLoadTest, BuildsLocalRankLookupFromSortedGlobalRanks) {
  constexpr char kConfig[] = R"json(
{
  "graph_name": "rank_lookup",
  "nodes": [
    {
      "name": "node0",
      "node_type": "vlm",
      "ranks": [4, 2, 3],
      "request_adapter": "request",
      "result_adapter": "result",
      "endpoint": {
        "transport": "in_process",
        "target": "node0"
      },
      "final_output": true,
      "output_keys": ["embeds"]
    }
  ]
}
)json";
  GraphConfig config = load_config_or_die(kConfig);

  ASSERT_EQ(config.nodes.size(), 1);
  const NodeConfig& node = config.nodes[0];
  ASSERT_EQ(node.ranks.size(), 3);
  EXPECT_EQ(node.ranks.at(2), 0);
  EXPECT_EQ(node.ranks.at(3), 1);
  EXPECT_EQ(node.ranks.at(4), 2);
}

}  // namespace
}  // namespace ensemble
}  // namespace xllm
