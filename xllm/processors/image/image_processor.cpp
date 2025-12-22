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

#include "image_processor.h"

namespace xllm {

torch::Tensor ImageProcessor::resize(const torch::Tensor& image,
                                     const std::vector<int64_t>& size,
                                     int resample,
                                     bool antialias) {
  LOG(INFO) << "resize input " << image.sizes();
  // if (image.dim() != 3) {
  //   LOG(FATAL) << "Input image must be a 3D tensor (C x H x W).";
  // }
  auto options = torch::nn::functional::InterpolateFuncOptions()
                     .size(size)
                     .align_corners(false)
                     .antialias(antialias);
  switch (resample) {
    case 1:
      options.mode(torch::kNearest);
      break;
    case 2:
      options.mode(torch::kBilinear);
      break;
    case 3:
      options.mode(torch::kBicubic);
      break;
    default:
      LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
  }
  return torch::nn::functional::interpolate(image, options)
      .clamp(0, 255)
      .to(torch::kUInt8);
}

torch::Tensor ImageProcessor::centerCrop(const torch::Tensor& image,
                                         const std::pair<int, int>& cropSize) {
  // if (image.dim() != 3) {
  //   LOG(FATAL)
  //       << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  // }

  int cropHeight = cropSize.first;
  int cropWidth = cropSize.second;
  int origHeight = image.size(1);
  int origWidth = image.size(2);

  int top = (origHeight - cropHeight) / 2;
  int bottom = top + cropHeight;
  int left = (origWidth - cropWidth) / 2;
  int right = left + cropWidth;

  if (top >= 0 && bottom <= origHeight && left >= 0 && right <= origWidth) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(top, bottom),
                        torch::indexing::Slice(left, right)});
  }

  int newHeight = std::max(cropHeight, origHeight);
  int newWidth = std::max(cropWidth, origWidth);
  auto paddedImage =
      torch::zeros({image.size(0), newHeight, newWidth}, image.options());

  int topPad = (newHeight - origHeight + 1) / 2;
  int leftPad = (newWidth - origWidth + 1) / 2;

  paddedImage.index_put_({torch::indexing::Slice(),
                          torch::indexing::Slice(topPad, topPad + origHeight),
                          torch::indexing::Slice(leftPad, leftPad + origWidth)},
                         image);

  top = (newHeight - cropHeight) / 2;
  bottom = top + cropHeight;
  left = (newWidth - cropWidth) / 2;
  right = left + cropWidth;

  return paddedImage.index({torch::indexing::Slice(),
                            torch::indexing::Slice(top, bottom),
                            torch::indexing::Slice(left, right)});
}

torch::Tensor ImageProcessor::rescale(const torch::Tensor& image,
                                      double scale) {
  return image * scale;
}

torch::Tensor ImageProcessor::normalize(const torch::Tensor& image,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& std) {
  // if (image.dim() != 3) {
  //   LOG(FATAL)
  //       << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  // }
  // TODO: NCHW
  int numChannels = image.size(1);
  if (mean.size() != numChannels || std.size() != numChannels) {
    LOG(FATAL) << "Mean and std vectors must have the same number "
               << "of elements as the number of channels in the "
               << "image.";
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto m_tensor = torch::tensor(mean, options).reshape({-1, 1, 1});
  auto s_tensor = torch::tensor(std, options).reshape({-1, 1, 1});

  result = result.sub(m_tensor);
  return result.div_(s_tensor);
}

std::pair<absl::flat_hash_map<Shape, torch::Tensor>,
          std::vector<std::pair<Shape, size_t>>>
ImageProcessor::group_images_by_shape(
    const std::vector<torch::Tensor>& images) {
  absl::flat_hash_map<Shape, std::vector<torch::Tensor>> temp_groups;
  std::vector<std::pair<Shape, size_t>> grouped_index;
  absl::flat_hash_map<Shape, torch::Tensor> grouped_images;
  grouped_index.reserve(images.size());
  for (auto& img : images) {
    auto sizes = img.sizes();
    Shape shape = {img.size(-2), img.size(-1)};
    auto& group = temp_groups[shape];
    size_t pos_in_group = group.size();
    group.push_back(img);
    grouped_index.push_back(std::make_pair(shape, pos_in_group - 1));
  }

  for (auto& [shape, tensors] : temp_groups) {
    grouped_images[shape] = torch::stack(tensors, 0);
  }
  return std::make_pair(std::move(grouped_images), std::move(grouped_index));
}

std::vector<torch::Tensor> ImageProcessor::reorder_images(
    absl::flat_hash_map<Shape, torch::Tensor> grouped_images,  // b,c,h,w
    std::vector<std::pair<Shape, size_t>> grouped_indexes) {
  std::vector<torch::Tensor> ordered_images;
  int image_num = grouped_indexes.size();
  ordered_images.reserve(image_num);

  for (int i = 0; i < image_num; i++) {
    ordered_images.push_back(
        grouped_images[grouped_indexes[i].first][grouped_indexes[i].second]);
  }
  return ordered_images;
}

std::pair<absl::flat_hash_map<Shape, torch::Tensor>,
          std::vector<std::pair<Shape, size_t>>>
ImageProcessor::group_videos_by_shape(
    const std::vector<torch::Tensor>& videos) {
  absl::flat_hash_map<Shape, std::vector<torch::Tensor>> temp_groups;
  std::vector<std::pair<Shape, size_t>> grouped_index;
  absl::flat_hash_map<Shape, torch::Tensor> grouped_videos;
  grouped_index.reserve(videos.size());
  // video [TCHW]
  for (auto& video : videos) {
    Shape shape = {video.size(0), video.size(-2), video.size(-1)};

    auto& group = temp_groups[shape];
    size_t pos_in_group = group.size();
    group.push_back(video);
    grouped_index.push_back(std::make_pair(shape, pos_in_group - 1));
  }

  for (auto& [shape, tensors] : temp_groups) {
    grouped_videos[shape] = torch::stack(tensors, 0);
  }
  return std::make_pair(std::move(grouped_videos), std::move(grouped_index));
}

std::vector<torch::Tensor> ImageProcessor::reorder_videos(
    absl::flat_hash_map<Shape, torch::Tensor> grouped_videos,
    std::vector<std::pair<Shape, size_t>> grouped_indexes) {
  std::vector<torch::Tensor> ordered_videos;
  int video_num = grouped_indexes.size();
  ordered_videos.reserve(video_num);

  for (int idx = 0; idx < video_num; idx++) {
    ordered_videos.push_back(grouped_videos[grouped_indexes[idx].first]
                                           [grouped_indexes[idx].second]);
  }
  return ordered_videos;
}

}  // namespace xllm