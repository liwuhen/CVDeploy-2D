cmake_minimum_required(VERSION 3.10.2)

option(ENABLE_CROSSCOMPILE "Enable crosscomplile for arm platform" OFF)
set(MODEL_FLAG
    "default"
    CACHE STRING "Model configuration flag")

add_definitions("-Wall -g")
add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(${CMAKE_CURRENT_LIST_DIR}/cmake/third_library.cmake)

if(${ENABLE_CROSSCOMPILE})
  include(${CMAKE_PATH}/platforms/linux/arm-toolchain.cmake)
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h.in"
               "${CMAKE_CURRENT_LIST_DIR}/modules/common/utils/std_cmake.h")

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/modules
  ${CMAKE_CURRENT_LIST_DIR}/modules/app_yolo
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/utils
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/logging
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_img
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_enum
  ${CMAKE_CURRENT_LIST_DIR}/modules/common/msg_struct)

add_subdirectory(modules/app_yolo)
add_subdirectory(runapp/${MODEL_FLAG})
