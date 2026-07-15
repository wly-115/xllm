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

namespace xllm {

inline constexpr char kPayloadFieldImage[] = "image";
inline constexpr char kPayloadFieldConditionImage[] = "condition_image";
inline constexpr char kPayloadFieldPrompt[] = "prompt";
inline constexpr char kPayloadFieldPromptTokens[] = "prompt_tokens";
inline constexpr char kPayloadFieldPromptMmData[] = "prompt_mm_data";
inline constexpr char kPayloadFieldNegativePrompt[] = "negative_prompt";
inline constexpr char kPayloadFieldNegativePromptTokens[] =
    "negative_prompt_tokens";
inline constexpr char kPayloadFieldNegativePromptMmData[] =
    "negative_prompt_mm_data";
inline constexpr char kPayloadFieldPromptEmbed[] = "prompt_embed";
inline constexpr char kPayloadFieldNegativePromptEmbed[] =
    "negative_prompt_embed";
inline constexpr char kPayloadFieldWidth[] = "width";
inline constexpr char kPayloadFieldHeight[] = "height";
inline constexpr char kPayloadFieldSeed[] = "seed";

}  // namespace xllm
