/* Copyright 2025-2026 The xLLM Authors.

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

#include "api_service/image_generation_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <string>
#include <utility>
#include <vector>

#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/omni_master.h"
#include "core/framework/request/dit_request_output.h"
#include "core/framework/request/dit_request_params.h"

namespace xllm {
namespace {

bool send_result_to_client_brpc(std::shared_ptr<ImageGenerationCall> call,
                                const std::string& request_id,
                                int64_t created_time,
                                const std::string& model,
                                const DiTRequestOutput& req_output) {
  auto& response = call->response();
  response.set_object("list");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);
  auto* proto_output = response.mutable_output();
  const std::vector<DiTGenerationOutput>& outputs = req_output.outputs;
  proto_output->mutable_results()->Reserve(outputs.size());

  for (const auto& output : outputs) {
    auto* proto_result = proto_output->add_results();
    proto_result->set_image(output.image);
    proto_result->set_width(output.width);
    proto_result->set_height(output.height);
    proto_result->set_seed(output.seed);
  }
  call->set_bytes_to_base64(true);
  return call->write_and_finish(response);
}

}  // namespace

ImageGenerationServiceImpl::ImageGenerationServiceImpl(
    DiTMaster* dit_master,
    OmniMaster* omni_master,
    const std::vector<std::string>& models)
    : APIServiceImpl(models),
      dit_master_(dit_master),
      omni_master_(omni_master) {
  CHECK(dit_master_ != nullptr || omni_master_ != nullptr)
      << "dit master and omni master cannot both be null.";
}

// image_generation_async for brpc
void ImageGenerationServiceImpl::process_async_impl(
    std::shared_ptr<ImageGenerationCall> call) {
  const auto& rpc_request = call->request();
  // check if model is supported
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  // Check if the request is being rate-limited.
  if (dit_master_ != nullptr && dit_master_->get_rate_limiter()->is_limited()) {
    call->finish_with_error(
        StatusCode::RESOURCE_EXHAUSTED,
        "The number of concurrent requests has reached the limit.");
    return;
  }

  // create DiTRequestParams for image generation request
  DiTRequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());

  std::string saved_request_id = request_params.request_id;
  if (omni_master_ != nullptr) {
    OmniRequestInput omni_input;
    omni_input.prompt = std::move(request_params.input_params.prompt);
    omni_input.negative_prompt =
        std::move(request_params.input_params.negative_prompt);
    omni_input.image = std::move(request_params.input_params.image);

    OmniRequestParams omni_params;
    omni_params.dit_generation_params =
        std::move(request_params.generation_params);

    omni_master_->handle_request(
        std::move(omni_input),
        std::move(omni_params),
        [call,
         model,
         request_id = std::move(saved_request_id),
         created_time = absl::ToUnixSeconds(absl::Now())](
            DiTRequestOutput req_output) -> bool {
          if (req_output.status.has_value()) {
            const Status& status = req_output.status.value();
            if (!status.ok()) {
              return call->finish_with_error(status.code(), status.message());
            }
          }

          return send_result_to_client_brpc(
              call, request_id, created_time, model, req_output);
        });
    return;
  }

  CHECK(dit_master_ != nullptr) << "dit master cannot be null.";
  // schedule the request
  dit_master_->handle_request(
      std::move(request_params),
      call.get(),
      [call,
       model,
       master = dit_master_,
       request_id = std::move(saved_request_id),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const DiTRequestOutput& req_output) -> bool {
        master->get_rate_limiter()->decrease_one_request();
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call->finish_with_error(status.code(), status.message());
          }
        }

        return send_result_to_client_brpc(
            call, request_id, created_time, model, req_output);
      });
}

}  // namespace xllm
