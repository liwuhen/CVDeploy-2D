cmake_minimum_required(VERSION 3.10.2)
project(${MODEL_FLAG}_project_exe)

include_directories(
  ${CMAKE_PATH}/modules/app_yolo
  ${CMAKE_PATH}/modules/app_yolo/appinterface
  ${CMAKE_PATH}/modules/app_yolo/architecture
  ${CMAKE_PATH}/modules/benchmark
  ${CMAKE_PATH}/modules/benchmark/voc
  ${CMAKE_PATH}/modules/benchmark/coco
  ${CMAKE_PATH}/modules/common/json
  ${CMAKE_PATH}/modules/common/utils
  ${CMAKE_PATH}/modules/common/logging
  ${CMAKE_PATH}/modules/common/msg_struct)

file(GLOB_RECURSE CPP_SRCS ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${CPP_SRCS})

target_link_libraries(${PROJECT_NAME} ${MODEL_FLAG}_project)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgcodecs
                      opencv_imgproc opencv_highgui)

set(EXECUTABLE_OUTPUT_PATH
    ${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG})

file(
  COPY ${CMAKE_PATH}/modules/config/data
  DESTINATION
    ${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/config)
file(
  COPY ${CMAKE_PATH}/modules/config/yaml/${MODEL_FLAG}/
  DESTINATION
    ${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/config/yaml)
file(
  COPY ${CMAKE_PATH}/modules/config/model/${MODEL_FLAG}/
  DESTINATION
    ${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPILER_DIR_FLAG}/config/model)
