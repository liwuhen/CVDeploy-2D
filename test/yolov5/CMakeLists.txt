cmake_minimum_required(VERSION 3.10.2)
project(${MODEL_FLAG}_project_exe)


include_directories(
    ${CMAKE_PATH}/modules/app_yolo/yolov5
    ${CMAKE_PATH}/modules/app_yolo/yolov5/appinterface
    ${CMAKE_PATH}/modules/app_yolo/yolov5/architecture
    ${CMAKE_PATH}/modules/common/utils
    ${CMAKE_PATH}/modules/common/logging
    ${CMAKE_PATH}/modules/common/msg_struct
)

file(GLOB_RECURSE CPP_SRCS
  ${PROJECT_SOURCE_DIR}/*.cpp 
)

add_executable(${PROJECT_NAME} ${CPP_SRCS})

target_link_libraries(${PROJECT_NAME} ${MODEL_FLAG}_project)
target_link_libraries(${PROJECT_NAME} 
  opencv_core
  opencv_imgcodecs
  opencv_imgproc
  opencv_highgui)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/${MODEL_FLAG}_bin/${COMPLIER_DIR_FLAG})

