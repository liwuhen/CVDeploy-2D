

if ( ${ENABLE_CROSSCOMPILE} )
  set(COMPLIER_DIR_FLAG arm)
  set(COMPLIER_FLAG aarch64_toolchain)
  set(CUDNN_DIR    "/home/aarch64_toolchain/cudnn")
  set(CUDA_TOOLKIT_ROOT_DIR "/home/aarch64_toolchain/cuda")
else()
  set(COMPLIER_DIR_FLAG x86)
  set(COMPLIER_FLAG x86_toolchain)
  set(CUDNN_DIR    "/usr/local/cuda-11.4")
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4/targets/x86_64-linux")
endif()

set(GLOG_DIR "/home/${COMPLIER_FLAG}/glog0.6.0")
set(EIGIN_DIR "/home/${COMPLIER_FLAG}/eigen3.4")
set(GFLAGS_DIR "/home/${COMPLIER_FLAG}/gflags_2.2.2")
set(OPENCV_DIR "/home/${COMPLIER_FLAG}/opencv3.4.5")
set(YAMLCPP_DIR "/home/${COMPLIER_FLAG}/yaml_cpp")
set(TENSORRT_DIR "/home/${COMPLIER_FLAG}/tensorrt")


include_directories(${GLOG_DIR}/include
                    ${EIGIN_DIR}/include
                    ${GFLAGS_DIR}/include
                    ${OPENCV_DIR}/include
                    ${YAMLCPP_DIR}/include
                    ${TENSORRT_DIR}/include
                    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

link_directories(${GLOG_DIR}/lib
                 ${GFLAGS_DIR}/lib
                 ${OPENCV_DIR}/lib
                 ${YAMLCPP_DIR}/lib
                 ${TENSORRT_DIR}/lib/stubs
                 ${CUDA_TOOLKIT_ROOT_DIR}/lib
)