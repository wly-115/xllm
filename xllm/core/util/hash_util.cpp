#include "hash_util.h"

#include <MurmurHash3.h>

#include "core/common/global_flags.h"

namespace xllm {

Murmur3Key hash_tensor(const torch::Tensor& tensor) {
  Murmur3Key key;
  torch::Tensor contiguous_tensor = tensor.contiguous();
  MurmurHash3_x86_128(
      reinterpret_cast<const void*>(contiguous_tensor.data_ptr()),
      tensor.numel() * tensor.element_size(),
      FLAGS_murmur_hash3_seed,
      key.data);
  return key;
}

Murmur3Key hash_string(const std::string& str) {
  Murmur3Key key;
  MurmurHash3_x86_128(reinterpret_cast<const void*>(str.data()),
                      str.size(),
                      FLAGS_murmur_hash3_seed,
                      key.data);
  return key;
}
}  // namespace xllm