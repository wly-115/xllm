include(cc_binary)

cc_binary(
  NAME 
    image_processor
  HDRS
    image_processor.h
    clip_image_processor.h
    minicpmv_image_processor.h
    qwen2_vl_image_processor.h
    glm4v_image_processor.h
  SRCS
    image_processor.cpp
    clip_image_processor.cpp
    minicpmv_image_processor.cpp
    qwen2_vl_image_processor.cpp
    glm4v_image_processor.cpp
  DEPS
    :models
    :model
    :distributed_runtime
    absl::strings
    xllm_kernels
    ascendcl
    nnopbase
    atb
    atb_customize
    c_sec
    spdlog::spdlog
)

add_dependencies(export_module spawn_worker)