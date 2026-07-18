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

#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <utility>
#include <vector>

#include "core/distributed_runtime/engine_data_adapter.h"
#include "core/distributed_runtime/engine_service_base.h"

namespace xllm {
namespace detail {

// The engine output callback lives in different places on the two request
// states: Request exposes a public output_func member, while DiTRequest keeps
// it private behind an accessor. Overload the install to keep the service
// template agnostic to that difference.
template <typename OutputFn>
void install_output_callback(Request& request, OutputFn&& output_fn) {
  request.state().output_func = std::forward<OutputFn>(output_fn);
}

template <typename OutputFn>
void install_output_callback(DiTRequest& request, OutputFn&& output_fn) {
  request.state().output_func() = std::forward<OutputFn>(output_fn);
}

}  // namespace detail

template <typename EngineType, typename RequestType, typename OutputType>
class EngineService final : public EngineServiceBase {
 public:
  EngineService(
      EngineType* engine,
      std::unique_ptr<EngineDataAdapter<RequestType, OutputType>> adapter,
      const NodeRuntimePlan& runtime_plan)
      : EngineServiceBase(runtime_plan),
        engine_(engine),
        adapter_(std::move(adapter)) {
    CHECK(engine_ != nullptr) << "Engine cannot be null.";
    CHECK(adapter_ != nullptr) << "Engine adapter cannot be null.";
  }

  ~EngineService() override { stop(); }

 private:
  EngineServiceExecutionResult execute(const NodeData& input) override {
    using OutputFuture = folly::SemiFuture<OutputType>;
    using OutputResult = folly::Try<OutputType>;
    using OutputResults = std::vector<OutputResult>;

    std::vector<std::shared_ptr<RequestType>> requests =
        adapter_->build_requests(input);
    std::vector<OutputFuture> futures;
    futures.reserve(requests.size());
    for (std::shared_ptr<RequestType>& request : requests) {
      std::shared_ptr<folly::Promise<OutputType>> promise =
          std::make_shared<folly::Promise<OutputType>>();
      futures.emplace_back(promise->getSemiFuture());
      detail::install_output_callback(
          *request,
          [promise = std::move(promise)](const OutputType& output) -> bool {
            promise->setValue(output);
            return true;
          });

      if (!engine_->add_request(request)) {
        return make_failure_result(
            NodeStatusCode::EXECUTION_FAILED,
            "No available resources to schedule engine request.");
      }
    }

    const std::chrono::system_clock::time_point deadline(
        std::chrono::milliseconds(input.context.deadline_ms));
    const folly::HighResDuration remaining_time =
        std::chrono::duration_cast<folly::HighResDuration>(
            deadline - std::chrono::system_clock::now());
    folly::Try<OutputResults> collected_outputs =
        folly::collectAll(futures).within(remaining_time).getTry();
    if (collected_outputs.template hasException<folly::FutureTimeout>()) {
      return make_failure_result(NodeStatusCode::TIMEOUT,
                                 "Timed out waiting for engine output.");
    }

    OutputResults output_results = std::move(collected_outputs).value();
    std::vector<OutputType> outputs;
    outputs.reserve(output_results.size());
    for (OutputResult& output_result : output_results) {
      outputs.emplace_back(std::move(output_result).value());
    }

    NodeData output = adapter_->convert_output(input, outputs);

    EngineServiceExecutionResult result;
    result.status.code = NodeStatusCode::OK;
    result.output = std::move(output);
    return result;
  }

  EngineType* engine_ = nullptr;
  std::unique_ptr<EngineDataAdapter<RequestType, OutputType>> adapter_;
};

}  // namespace xllm
