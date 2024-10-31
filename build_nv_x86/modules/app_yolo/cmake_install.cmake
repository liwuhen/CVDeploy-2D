# Install script for directory: /home/IM_Work/HPC_Deploy/modules/app_yolo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs" TYPE SHARED_LIBRARY FILES "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so")
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so"
         OLD_RPATH "/home/x86_toolchain/glog0.6.0/lib:/home/x86_toolchain/gflags_2.2.2/lib:/home/x86_toolchain/opencv3.4.5/lib:/home/x86_toolchain/yaml_cpp/lib:/home/x86_toolchain/tensorrt/lib/stubs:/usr/local/cuda-11.4/targets/x86_64-linux/lib:/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/libs/libyolov5_project.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor" TYPE SHARED_LIBRARY FILES "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_decode.so")
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so"
         OLD_RPATH "/home/x86_toolchain/glog0.6.0/lib:/home/x86_toolchain/gflags_2.2.2/lib:/home/x86_toolchain/opencv3.4.5/lib:/home/x86_toolchain/yaml_cpp/lib:/home/x86_toolchain/tensorrt/lib/stubs:/usr/local/cuda-11.4/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_decode.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor" TYPE SHARED_LIBRARY FILES "/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/proprecessor/libyolov5_gpu_proprecessor.so")
  if(EXISTS "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so"
         OLD_RPATH "/home/x86_toolchain/glog0.6.0/lib:/home/x86_toolchain/gflags_2.2.2/lib:/home/x86_toolchain/opencv3.4.5/lib:/home/x86_toolchain/yaml_cpp/lib:/home/x86_toolchain/tensorrt/lib/stubs:/usr/local/cuda-11.4/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/IM_Work/HPC_Deploy/modules/app_yolo/yolov5_bin/x86/postprecessor/libyolov5_gpu_proprecessor.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

