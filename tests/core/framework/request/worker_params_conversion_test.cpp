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

#include "core/framework/request/worker_params_conversion.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(WorkerParamsConversionTest, SamplingParamsProtoRoundTrip) {
  RequestSamplingParam params;
  params.frequency_penalty = 0.25f;
  params.presence_penalty = 0.5f;
  params.repetition_penalty = 1.25f;
  params.temperature = 0.75f;
  params.top_p = 0.875f;
  params.top_k = 64;
  params.logprobs = true;
  params.top_logprobs = 8;
  params.do_sample = true;
  params.is_embeddings = true;
  params.beam_width = 4;
  params.num_return_sequences = 3;

  proto::RequestSamplingParam proto_params;
  sampling_params_to_proto(params, &proto_params);
  RequestSamplingParam round_trip;
  sampling_params_from_proto(proto_params, &round_trip);

  EXPECT_FLOAT_EQ(round_trip.frequency_penalty, params.frequency_penalty);
  EXPECT_FLOAT_EQ(round_trip.presence_penalty, params.presence_penalty);
  EXPECT_FLOAT_EQ(round_trip.repetition_penalty, params.repetition_penalty);
  EXPECT_FLOAT_EQ(round_trip.temperature, params.temperature);
  EXPECT_FLOAT_EQ(round_trip.top_p, params.top_p);
  EXPECT_EQ(round_trip.top_k, params.top_k);
  EXPECT_EQ(round_trip.logprobs, params.logprobs);
  EXPECT_EQ(round_trip.top_logprobs, params.top_logprobs);
  EXPECT_EQ(round_trip.do_sample, params.do_sample);
  EXPECT_EQ(round_trip.is_embeddings, params.is_embeddings);
  EXPECT_EQ(round_trip.beam_width, params.beam_width);
  EXPECT_EQ(round_trip.num_return_sequences, params.num_return_sequences);
}

TEST(WorkerParamsConversionTest, DiTGenerationParamsProtoRoundTrip) {
  DiTGenerationParams params;
  params.width = 1024;
  params.height = 768;
  params.num_inference_steps = 42;
  params.true_cfg_scale = 1.25f;
  params.guidance_scale = 4.5f;
  params.num_images_per_prompt = 2;
  params.seed = 123456789;
  params.max_sequence_length = 1024;
  params.strength = 0.625f;
  params.enable_cfg_renorm = false;
  params.cfg_renorm_min = 0.125f;
  params.audio_duration_frames = 512;
  params.audio_steps = 24;
  params.audio_guidance_method = "apg";
  params.audio_sampling_rate = 48000;
  params.num_videos_per_prompt = 3;
  params.num_frames = 97;
  params.force_video_output = true;
  params.video_fps = 12.5;
  params.guidance_scale_2 = 2.25f;
  params.seconds = 8;
  params.boundary_ratio = 0.75f;
  params.flow_shift = 3.0f;

  proto::DiTGenerationParams proto_params;
  dit_generation_params_to_proto(params, &proto_params);
  DiTGenerationParams round_trip;
  dit_generation_params_from_proto(proto_params, &round_trip);

  EXPECT_EQ(round_trip, params);
}

}  // namespace
}  // namespace xllm
