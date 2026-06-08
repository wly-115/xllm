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

#include <string>

#include "omni_request_state.h"
#include "request_base.h"

namespace xllm {

class OmniRequest final : public RequestBase {
 public:
  OmniRequest(const std::string& request_id,
              const std::string& x_request_id,
              const std::string& x_request_time,
              OmniRequestState state,
              const std::string& service_request_id = "",
              const std::string& source_xservice_addr = "")
      : RequestBase(request_id,
                    x_request_id,
                    x_request_time,
                    service_request_id,
                    source_xservice_addr),
        state_(std::move(state)) {}

  OmniRequestState& state() { return state_; }
  const OmniRequestState& state() const { return state_; }

 private:
  OmniRequestState state_;
};

}  // namespace xllm
