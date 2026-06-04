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

#include "core/framework/ensemble/engine_config.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/options.h"
#include "core/util/device_name_utils.h"
#include "core/util/net.h"

namespace xllm {
namespace ensemble {
namespace {

void apply_engine_config(
    const std::unordered_map<std::string, std::string>& engine_config) {
  for (const auto& [name, value] : engine_config) {
    const std::string applied_value =
        google::SetCommandLineOption(name.c_str(), value.c_str());
  }
}

const NodeConfig& find_node_config(const GraphConfig& config,
                                   int32_t global_rank) {
  for (const NodeConfig& candidate : config.nodes) {
    if (candidate.ranks.find(global_rank) != candidate.ranks.end()) {
      return candidate;
    }
  }
  LOG(FATAL) << "graph global rank does not belong to any node: "
             << global_rank;
}

Options build_options(int32_t local_rank, int32_t world_size) {
  Options options;
  const int32_t max_tokens_per_chunk_for_prefill =
      FLAGS_max_tokens_per_chunk_for_prefill < 0
          ? FLAGS_max_tokens_per_batch
          : FLAGS_max_tokens_per_chunk_for_prefill;
  const bool is_local = !FLAGS_host.empty() &&
                        net::extract_ip(FLAGS_master_node_addr) == FLAGS_host;
#if defined(USE_NPU)
  options.npu_kernel_backend(FLAGS_npu_kernel_backend);
#endif
  options.model_path(FLAGS_model)
      .model_id(FLAGS_model_id)
      .task_type(FLAGS_task)
      .devices(FLAGS_devices)
      .draft_model_path(FLAGS_draft_model)
      .draft_devices(FLAGS_draft_devices)
      .backend(FLAGS_backend)
      .limit_image_per_prompt(FLAGS_limit_image_per_prompt)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .max_tokens_per_batch(FLAGS_max_tokens_per_batch)
      .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
      .max_tokens_per_chunk_for_prefill(max_tokens_per_chunk_for_prefill)
      .num_speculative_tokens(FLAGS_num_speculative_tokens)
      .speculative_algorithm(FLAGS_speculative_algorithm)
      .speculative_suffix_cache_max_depth(
          FLAGS_speculative_suffix_cache_max_depth)
      .speculative_suffix_max_spec_factor(
          FLAGS_speculative_suffix_max_spec_factor)
      .speculative_suffix_max_spec_offset(
          FLAGS_speculative_suffix_max_spec_offset)
      .speculative_suffix_min_token_prob(
          FLAGS_speculative_suffix_min_token_prob)
      .speculative_suffix_max_cached_requests(
          FLAGS_speculative_suffix_max_cached_requests)
      .speculative_suffix_use_tree_spec(FLAGS_speculative_suffix_use_tree_spec)
      .num_request_handling_threads(FLAGS_num_request_handling_threads)
      .communication_backend(FLAGS_communication_backend)
      .enable_eplb(FLAGS_enable_eplb)
      .redundant_experts_num(FLAGS_redundant_experts_num)
      .eplb_update_interval(FLAGS_eplb_update_interval)
      .eplb_update_threshold(FLAGS_eplb_update_threshold)
      .rank_tablefile(FLAGS_rank_tablefile)
      .expert_parallel_degree(FLAGS_expert_parallel_degree)
      .enable_chunked_prefill(FLAGS_enable_chunked_prefill)
      .enable_prefill_sp(FLAGS_enable_prefill_sp)
      .master_node_addr(FLAGS_master_node_addr)
      .instance_role(InstanceRole(FLAGS_instance_role))
      .device_ip("")
      .transfer_listen_port(static_cast<uint16_t>(FLAGS_transfer_listen_port))
      .nnodes(world_size)
      .node_rank(local_rank)
      .dp_size(FLAGS_dp_size)
      .cp_size(FLAGS_cp_size)
      .ep_size(FLAGS_ep_size)
      .tp_size(FLAGS_tp_size)
      .sp_size(FLAGS_sp_size)
      .cfg_size(FLAGS_cfg_size)
      .instance_name(FLAGS_host + ":" + std::to_string(FLAGS_port))
      .enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_pd_ooc(FLAGS_enable_pd_ooc)
      .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
      .enable_service_routing(FLAGS_enable_service_routing ||
                              FLAGS_enable_disagg_pd)
      .tool_call_parser(FLAGS_tool_call_parser)
      .reasoning_parser(FLAGS_reasoning_parser)
      .priority_strategy(FLAGS_priority_strategy)
      .enable_online_preempt_offline(FLAGS_enable_online_preempt_offline)
      .enable_cache_upload(
          (FLAGS_enable_service_routing || FLAGS_enable_disagg_pd) &&
          FLAGS_enable_prefix_cache && FLAGS_enable_cache_upload)
      .host_blocks_factor(FLAGS_host_blocks_factor)
      .enable_kvcache_store(FLAGS_enable_kvcache_store &&
                            FLAGS_enable_prefix_cache &&
                            (FLAGS_host_blocks_factor > 1.0))
      .prefetch_timeout(FLAGS_prefetch_timeout)
      .prefetch_bacth_size(FLAGS_prefetch_bacth_size)
      .layers_wise_copy_batchs(FLAGS_layers_wise_copy_batchs)
      .store_protocol(FLAGS_store_protocol)
      .store_master_server_address(FLAGS_store_master_server_address)
      .store_metadata_server(FLAGS_store_metadata_server)
      .store_local_hostname(FLAGS_store_local_hostname)
      .enable_multi_stream_parallel(FLAGS_enable_multi_stream_parallel)
      .enable_profile_step_time(FLAGS_enable_profile_step_time)
      .enable_profile_token_budget(FLAGS_enable_profile_token_budget)
      .enable_latency_aware_schedule(FLAGS_enable_latency_aware_schedule)
      .profile_max_prompt_length(FLAGS_profile_max_prompt_length)
      .enable_profile_kv_blocks(FLAGS_enable_profile_kv_blocks)
      .disable_ttft_profiling(FLAGS_disable_ttft_profiling)
      .enable_forward_interruption(FLAGS_enable_forward_interruption)
      .enable_graph(FLAGS_enable_graph)
      .max_global_ttft_ms(FLAGS_max_global_ttft_ms)
      .max_global_tpot_ms(FLAGS_max_global_tpot_ms)
      .max_requests_per_batch(FLAGS_max_requests_per_batch)
      .enable_shm(FLAGS_enable_shm)
      .input_shm_size(FLAGS_input_shm_size)
      .output_shm_size(FLAGS_output_shm_size)
      .beam_width(FLAGS_beam_width)
      .kv_cache_dtype(FLAGS_kv_cache_dtype)
      .rec_worker_max_concurrency(
          static_cast<int32_t>(FLAGS_rec_worker_max_concurrency))
      .is_local(is_local);
  return options;
}

runtime::Options build_runtime_options(const Options& options) {
  const int32_t local_rank = options.node_rank();
  const int32_t world_size = options.nnodes();

  runtime::Options runtime_options;
#if defined(USE_NPU)
  runtime_options.npu_kernel_backend(options.npu_kernel_backend());
#endif
  runtime_options.model_path(options.model_path())
      .model_id(options.model_id())
      .draft_model_path(options.draft_model_path())
      .devices(DeviceNameUtils::parse_devices(
          options.devices().value_or(FLAGS_devices)))
      .draft_devices(DeviceNameUtils::parse_devices(
          options.draft_devices().value_or(FLAGS_draft_devices)))
      .backend(options.backend())
      .block_size(options.block_size())
      .max_cache_size(options.max_cache_size())
      .max_memory_utilization(options.max_memory_utilization())
      .enable_prefix_cache(options.enable_prefix_cache())
      .num_speculative_tokens(options.num_speculative_tokens())
      .speculative_algorithm(options.speculative_algorithm())
      .speculative_suffix_cache_max_depth(
          options.speculative_suffix_cache_max_depth())
      .speculative_suffix_max_spec_factor(
          options.speculative_suffix_max_spec_factor())
      .speculative_suffix_max_spec_offset(
          options.speculative_suffix_max_spec_offset())
      .speculative_suffix_min_token_prob(
          options.speculative_suffix_min_token_prob())
      .speculative_suffix_max_cached_requests(
          options.speculative_suffix_max_cached_requests())
      .speculative_suffix_use_tree_spec(
          options.speculative_suffix_use_tree_spec())
      .world_size(world_size)
      .task_type(options.task_type())
      .enable_mla(options.enable_mla())
      .master_node_addr(options.master_node_addr())
      .nnodes(world_size)
      .node_rank(local_rank)
      .dp_size(options.dp_size())
      .ep_size(options.ep_size())
      .cp_size(options.cp_size())
      .tp_size(options.tp_size())
      .sp_size(options.sp_size())
      .cfg_size(options.cfg_size())
      .enable_schedule_overlap(options.enable_schedule_overlap())
      .enable_chunked_prefill(options.enable_chunked_prefill())
      .enable_prefill_sp(options.enable_prefill_sp())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .max_tokens_per_batch(options.max_tokens_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .instance_name(options.instance_name())
      .enable_disagg_pd(options.enable_disagg_pd())
      .enable_pd_ooc(options.enable_pd_ooc())
      .instance_role(options.instance_role())
      .kv_cache_transfer_mode(options.kv_cache_transfer_mode())
      .transfer_listen_port(options.transfer_listen_port())
      .enable_service_routing(options.enable_service_routing())
      .disable_ttft_profiling(options.disable_ttft_profiling())
      .enable_forward_interruption(options.enable_forward_interruption())
      .priority_strategy(options.priority_strategy())
      .enable_online_preempt_offline(options.enable_online_preempt_offline())
      .enable_cache_upload(options.enable_cache_upload())
      .host_blocks_factor(options.host_blocks_factor())
      .enable_kvcache_store(options.enable_kvcache_store())
      .store_protocol(options.store_protocol())
      .store_master_server_address(options.store_master_server_address())
      .store_metadata_server(options.store_metadata_server())
      .store_local_hostname(options.store_local_hostname())
      .prefetch_bacth_size(options.prefetch_bacth_size())
      .layers_wise_copy_batchs(options.layers_wise_copy_batchs())
      .max_requests_per_batch(options.max_requests_per_batch())
      .enable_shm(options.enable_shm())
      .input_shm_size(options.input_shm_size())
      .output_shm_size(options.output_shm_size())
      .is_local(options.is_local())
      .enable_graph(options.enable_graph())
      .beam_width(options.beam_width())
      .kv_cache_dtype(options.kv_cache_dtype())
      .rec_worker_max_concurrency(options.rec_worker_max_concurrency());
  return runtime_options;
}

}  // namespace

NodeLaunchConfig resolve_node_launch_config(const GraphConfig& config,
                                            int32_t global_rank) {
  const NodeConfig& node = find_node_config(config, global_rank);
  apply_engine_config(node.engine_config);

  const int32_t local_rank = node.ranks.at(global_rank);
  const int32_t world_size = static_cast<int32_t>(node.ranks.size());
  Options options = build_options(local_rank, world_size);
  NodeLaunchConfig launch_config;
  launch_config.node_name = node.name;
  launch_config.service_target = node.endpoint.target;
  launch_config.runtime_options = build_runtime_options(options);
  launch_config.graph_global_rank = global_rank;
  launch_config.local_rank = local_rank;
  launch_config.world_size = world_size;
  launch_config.is_leader = node.ranks.begin()->first == global_rank;
  return launch_config;
}

}  // namespace ensemble
}  // namespace xllm
