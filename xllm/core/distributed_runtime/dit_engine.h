/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <absl/time/time.h>
#include <gflags/gflags.h>

#include <memory>

#include "common/macros.h"
#include "dist_manager.h"
#include "engine.h"
#include "framework/batch/dit_batch.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "runtime/dit_worker_impl.h"

namespace xllm {
class DiTRequest;
class DiTScheduler;

class DiTEngine : public Engine {
 public:
  DiTEngine(const runtime::Options& options,
            std::shared_ptr<DistManager> dist_manager = nullptr);

  ~DiTEngine();

  DiTForwardOutput step(std::vector<DiTBatch>& batch);

  const runtime::Options& options() const { return options_; }

  bool init();

  bool init_scheduler() override;

  bool add_request(std::shared_ptr<DiTRequest>& request);

  void incr_pending_requests(size_t count) override;

  void decr_pending_requests() override;

  void step_scheduler(const absl::Duration& timeout) override;

  void generate() override;

  // return the active activation memory
  std::vector<int64_t> get_active_activation_memory() const;

  std::shared_ptr<DistManager> get_dist_manager() { return dist_manager_; }

  // These two functions wouldn't be used in dit inference progress
  ForwardOutput step(std::vector<Batch>& batch) override {
    ForwardOutput output;
    return output;
  }

  void update_last_step_result(std::vector<Batch>& batch) override { return; }

 protected:
  // worker client which is used for call worker
  // The reason for adding a worker client is to unify the
  // access code for both local and remote workers, thereby
  // introducing an additional worker_client abstraction.
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;

  // For multi-node serving
  // engine brpc server, all workers connect to engine_server_,
  // engine_server_ will send a UniqueId for workers to
  // create process group. And workers send worker brpc server
  // address to engine, engine will create WorkerClient for each worker.
  // Engine call workers to step via these WorkerClients.
  std::shared_ptr<DistManager> dist_manager_ = nullptr;

  std::unique_ptr<ThreadPool> threadpool_ = nullptr;

 private:
  // setup workers internal
  void setup_workers(const runtime::Options& options);
  // init models
  bool init_model();
  // options
  runtime::Options options_;
  // num of worker_clients
  int64_t worker_clients_num_;
  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  std::unique_ptr<DiTScheduler> scheduler_;
};

}  // namespace xllm
