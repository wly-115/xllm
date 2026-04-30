include(cc_library)

add_subdirectory(common)
add_subdirectory(distributed_runtime)
add_subdirectory(framework)

cc_library(
  NAME
    omni_master
  HDRS
    distributed_runtime/omni_master.h
  SRCS
    distributed_runtime/omni_master.cpp
  DEPS
    ensemble_engine
    ensemble_node_ready_service
    :graph
    glog::glog
)

add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(platform)
add_subdirectory(runtime)
add_subdirectory(scheduler)
add_subdirectory(util)
