cmake_minimum_required(VERSION 3.10.2)
project(${MODEL_FLAG}_project)

set(LIBS "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/libs")
set(CONFIG_PATH "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/config")
set(THIRD_PARTY "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/third_party")
set(WORKSPACE_PATH "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/workspace")
set(PROCESSOR_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/proprecessor")
set(POSTPRECESSOR_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG}/postprecessor")

find_package(CUDA REQUIRED)

###########
## Build ##
###########

include_directories(
  ${PROJECT_SOURCE_DIR}/app
  ${PROJECT_SOURCE_DIR}/appinterface
  ${PROJECT_SOURCE_DIR}/architecture
  ${PROJECT_SOURCE_DIR}/architecture/cuda
  ${PROJECT_SOURCE_DIR}/architecture/common
  ${PROJECT_SOURCE_DIR}/architecture/module
  ${CMAKE_PATH}/modules/common/utils
  ${CMAKE_PATH}/modules/common/logging
  ${CMAKE_PATH}/modules/common/msg_img
  ${CMAKE_PATH}/modules/common/msg_enum
  ${CMAKE_PATH}/modules/common/msg_struct
)

link_libraries(${GLOG_DIR}/lib/libglog.a
               ${GFLAGS_DIR}/lib/libgflags.a)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS -O3;-G;-g; "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -G -w ${CUDA_GEN_CODE}")

file(GLOB_RECURSE CPP_SRCS
  ${PROJECT_SOURCE_DIR}/app/*.cpp
  ${PROJECT_SOURCE_DIR}/appinterface/*.cpp
  ${PROJECT_SOURCE_DIR}/architecture/*.cpp 
  ${PROJECT_SOURCE_DIR}/architecture/cuda/*.cpp 
  ${PROJECT_SOURCE_DIR}/architecture/common/*.cpp 
  ${CMAKE_PATH}/modules/common/logging/*.cpp 
) 

cuda_add_library(${MODEL_FLAG}_gpu_proprecessor
                 SHARED
                 ${PROJECT_SOURCE_DIR}/architecture/cuda/warpaffine.cu)

cuda_add_library(${MODEL_FLAG}_gpu_decode
                 SHARED
                 ${PROJECT_SOURCE_DIR}/architecture/cuda/gpu_decode.cu)

set_target_properties(${MODEL_FLAG}_gpu_proprecessor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROCESSOR_OUTPUT_DIR}"
    OUTPUT_NAME "${MODEL_FLAG}_gpu_proprecessor"
)

set_target_properties(${MODEL_FLAG}_gpu_decode PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${POSTPRECESSOR_OUTPUT_DIR}"
    OUTPUT_NAME "${MODEL_FLAG}_gpu_decode"
)

add_library(${PROJECT_NAME} SHARED ${CPP_SRCS})

target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(${PROJECT_NAME} ${MODEL_FLAG}_gpu_proprecessor)
target_link_libraries(${PROJECT_NAME} yaml-cpp ${EIGIN_DIR})
target_link_libraries(${PROJECT_NAME} 
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
  opencv_highgui)

set_target_properties(${PROJECT_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${LIBS}"
  OUTPUT_NAME "${PROJECT_NAME}"
)

include(${CMAKE_SOURCE_DIR}/cmake/install-toolchain.cmake)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${LIBS})
install(TARGETS ${MODEL_FLAG}_gpu_decode LIBRARY DESTINATION ${PROCESSOR_OUTPUT_DIR})
install(TARGETS ${MODEL_FLAG}_gpu_proprecessor LIBRARY DESTINATION ${POSTPRECESSOR_OUTPUT_DIR})

