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

#include "mm_input.h"

#include <glog/logging.h>

#include <atomic>

#include "core/util/blocking_counter.h"
#include "core/util/threadpool.h"
#include "mm_handler.h"

namespace xllm {
namespace {
struct DecodeTask {
  std::string type;
  size_t input_index;
};
}  // namespace

MMInputTransfer::MMInputTransfer() {
  mm_handlers_ = std::make_unique<MMHandlerSet>();
  threadpool_ = std::make_unique<ThreadPool>(/*num_threads=*/16);
}

MMInputTransfer::~MMInputTransfer() {}

MMErrCode MMInputTransfer::trans(const std::vector<Message>& messages,
                                 MMInput& inputs) {
  inputs.clear();
  std::vector<MMInputItem> ins;

  for (size_t idx = 0; idx < messages.size(); ++idx) {
    const auto& message = messages[idx];
    const auto& mmc = std::get<MMContentVec>(message.content);

    MMErrCode code = this->trans(mmc, ins, inputs.payload());
    if (code != MMErrCode::SUCCESS) {
      return code;
    }

    inputs.insert(ins);
  }
  return MMErrCode::SUCCESS;
}

MMErrCode MMInputTransfer::trans(const MMContentVec& mmc,
                                 std::vector<MMInputItem>& inputs,
                                 MMPayload& payload) {
  inputs.clear();
  inputs.reserve(mmc.size());
  std::vector<DecodeTask> decode_tasks;
  decode_tasks.reserve(mmc.size());

  for (size_t idx = 0; idx < mmc.size(); ++idx) {
    const auto& item = mmc[idx];
    const auto& type = item.type;
    if (type == "text") {
      continue;
    }
    MMInputItem input;
    MMErrCode code = mm_handlers_->load(type, item, input, payload);
    if (code != MMErrCode::SUCCESS) {
      return code;
    }

    inputs.emplace_back(std::move(input));
    decode_tasks.emplace_back(DecodeTask{
        .type = type,
        .input_index = inputs.size() - 1,
    });
  }

  if (decode_tasks.empty()) {
    return MMErrCode::SUCCESS;
  }

  if (decode_tasks.size() == 1) {
    const DecodeTask& decode_task = decode_tasks.front();
    MMErrCode code =
        mm_handlers_->decode(decode_task.type, inputs[decode_task.input_index]);
    if (code != MMErrCode::SUCCESS) {
      LOG(ERROR) << "decode failed at input index " << decode_task.input_index
                 << ", type=" << decode_task.type;
      return code;
    }
    return MMErrCode::SUCCESS;
  }
  int32_t task_count = static_cast<int32_t>(decode_tasks.size());
  BlockingCounter counter(task_count);
  std::atomic<MMErrCode> error(MMErrCode::SUCCESS);
  for (const DecodeTask& decode_task : decode_tasks) {
    threadpool_->schedule(
        [this, &counter, &error, &inputs, decode_task]() mutable {
          if (error.load() != MMErrCode::SUCCESS) {
            counter.decrement_count();
            return;
          }

          MMErrCode code = mm_handlers_->decode(
              decode_task.type, inputs[decode_task.input_index]);
          if (code != MMErrCode::SUCCESS) {
            LOG(ERROR) << "parallel decode failed at input index "
                       << decode_task.input_index
                       << ", type=" << decode_task.type;
            MMErrCode expected = MMErrCode::SUCCESS;
            error.compare_exchange_strong(expected, code);
          }
          counter.decrement_count();
        });
  }
  counter.wait();

  return error.load();
}

}  // namespace xllm
