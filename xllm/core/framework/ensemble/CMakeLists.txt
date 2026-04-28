include(cc_library)
include(cc_test)

cc_library(
  NAME
    graph_config
  HDRS
    graph_config.h
  SRCS
    graph_config.cpp
  DEPS
    :common
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    graph_config_test
  SRCS
    graph_config_test.cpp
  DEPS
    graph_config
    GTest::gtest_main
)
