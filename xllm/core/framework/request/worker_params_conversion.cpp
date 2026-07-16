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

#include <glog/logging.h>

namespace xllm {

void sampling_params_to_proto(
    const RequestSamplingParam& sampling_params,
    proto::RequestSamplingParam* proto_sampling_params) {
  CHECK(proto_sampling_params != nullptr)
      << "RequestSamplingParam proto output cannot be null.";
  proto_sampling_params->Clear();
  proto_sampling_params->set_frequency_penalty(
      sampling_params.frequency_penalty);
  proto_sampling_params->set_presence_penalty(sampling_params.presence_penalty);
  proto_sampling_params->set_repetition_penalty(
      sampling_params.repetition_penalty);
  proto_sampling_params->set_temperature(sampling_params.temperature);
  proto_sampling_params->set_top_p(sampling_params.top_p);
  proto_sampling_params->set_top_k(sampling_params.top_k);
  proto_sampling_params->set_logprobs(sampling_params.logprobs);
  proto_sampling_params->set_top_logprobs(sampling_params.top_logprobs);
  proto_sampling_params->set_do_sample(sampling_params.do_sample);
  proto_sampling_params->set_is_embeddings(sampling_params.is_embeddings);
  proto_sampling_params->set_beam_width(sampling_params.beam_width);
  proto_sampling_params->set_num_return_sequences(
      sampling_params.num_return_sequences);
}

void sampling_params_from_proto(
    const proto::RequestSamplingParam& proto_sampling_params,
    RequestSamplingParam* sampling_params) {
  CHECK(sampling_params != nullptr)
      << "RequestSamplingParam output cannot be null.";
  sampling_params->frequency_penalty =
      proto_sampling_params.frequency_penalty();
  sampling_params->presence_penalty = proto_sampling_params.presence_penalty();
  sampling_params->repetition_penalty =
      proto_sampling_params.repetition_penalty();
  sampling_params->temperature = proto_sampling_params.temperature();
  sampling_params->top_p = proto_sampling_params.top_p();
  sampling_params->top_k = proto_sampling_params.top_k();
  sampling_params->logprobs = proto_sampling_params.logprobs();
  sampling_params->top_logprobs = proto_sampling_params.top_logprobs();
  sampling_params->do_sample = proto_sampling_params.do_sample();
  sampling_params->is_embeddings = proto_sampling_params.is_embeddings();
  sampling_params->beam_width = proto_sampling_params.beam_width();
  sampling_params->num_return_sequences =
      proto_sampling_params.num_return_sequences();
}

void dit_generation_params_to_proto(
    const DiTGenerationParams& generation_params,
    proto::DiTGenerationParams* proto_generation_params) {
  CHECK(proto_generation_params != nullptr)
      << "DiTGenerationParams proto output cannot be null.";
  proto_generation_params->Clear();
  proto_generation_params->set_width(generation_params.width);
  proto_generation_params->set_height(generation_params.height);
  proto_generation_params->set_num_inference_steps(
      generation_params.num_inference_steps);
  proto_generation_params->set_true_cfg_scale(generation_params.true_cfg_scale);
  proto_generation_params->set_guidance_scale(generation_params.guidance_scale);
  proto_generation_params->set_num_images_per_prompt(
      generation_params.num_images_per_prompt);
  proto_generation_params->set_seed(generation_params.seed);
  proto_generation_params->set_max_sequence_length(
      generation_params.max_sequence_length);
  proto_generation_params->set_strength(generation_params.strength);
  proto_generation_params->set_enable_cfg_renorm(
      generation_params.enable_cfg_renorm);
  proto_generation_params->set_cfg_renorm_min(generation_params.cfg_renorm_min);
  proto_generation_params->set_audio_duration_frames(
      generation_params.audio_duration_frames);
  proto_generation_params->set_audio_steps(generation_params.audio_steps);
  proto_generation_params->set_audio_guidance_method(
      generation_params.audio_guidance_method);
  proto_generation_params->set_audio_sampling_rate(
      generation_params.audio_sampling_rate);
  proto_generation_params->set_num_videos_per_prompt(
      generation_params.num_videos_per_prompt);
  proto_generation_params->set_num_frames(generation_params.num_frames);
  proto_generation_params->set_force_video_output(
      generation_params.force_video_output);
  proto_generation_params->set_video_fps(generation_params.video_fps);
  proto_generation_params->set_guidance_scale_2(
      generation_params.guidance_scale_2);
  proto_generation_params->set_seconds(generation_params.seconds);
  proto_generation_params->set_boundary_ratio(generation_params.boundary_ratio);
  proto_generation_params->set_flow_shift(generation_params.flow_shift);
}

void dit_generation_params_from_proto(
    const proto::DiTGenerationParams& proto_generation_params,
    DiTGenerationParams* generation_params) {
  CHECK(generation_params != nullptr)
      << "DiTGenerationParams output cannot be null.";
  generation_params->width = proto_generation_params.width();
  generation_params->height = proto_generation_params.height();
  generation_params->num_inference_steps =
      proto_generation_params.num_inference_steps();
  generation_params->true_cfg_scale = proto_generation_params.true_cfg_scale();
  generation_params->guidance_scale = proto_generation_params.guidance_scale();
  generation_params->num_images_per_prompt =
      proto_generation_params.num_images_per_prompt();
  generation_params->seed = proto_generation_params.seed();
  generation_params->max_sequence_length =
      proto_generation_params.max_sequence_length();
  generation_params->strength = proto_generation_params.strength();
  generation_params->enable_cfg_renorm =
      proto_generation_params.enable_cfg_renorm();
  generation_params->cfg_renorm_min = proto_generation_params.cfg_renorm_min();
  generation_params->audio_duration_frames =
      proto_generation_params.audio_duration_frames();
  generation_params->audio_steps = proto_generation_params.audio_steps();
  generation_params->audio_guidance_method =
      proto_generation_params.audio_guidance_method();
  generation_params->audio_sampling_rate =
      proto_generation_params.audio_sampling_rate();
  generation_params->num_videos_per_prompt =
      proto_generation_params.num_videos_per_prompt();
  generation_params->num_frames = proto_generation_params.num_frames();
  generation_params->force_video_output =
      proto_generation_params.force_video_output();
  generation_params->video_fps = proto_generation_params.video_fps();
  generation_params->guidance_scale_2 =
      proto_generation_params.guidance_scale_2();
  generation_params->seconds = proto_generation_params.seconds();
  generation_params->boundary_ratio = proto_generation_params.boundary_ratio();
  generation_params->flow_shift = proto_generation_params.flow_shift();
}

}  // namespace xllm
