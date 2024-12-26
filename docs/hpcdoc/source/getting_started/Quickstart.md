# Quickstart
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white)

## 1. Build project
![docker](https://img.shields.io/badge/How%20to%20build-docker-brightgreen)
### 1.1 Configuring the arm version of the gcc and g++ cross-compile environments in a Docker environment

- In the **platforms/arm** directory, replace the following path in arm-toolchain.cmake with the path to the arm toolchain in the docker container (the specific path to the arm toolchain is up to you)

    ```shell
    # Setting up your computer's arm gcc,g++ environment -- arm version
    set(CMAKE_C_COMPILER
        /home/IM/aarch64_toolchain/aarch64_gun/bin/aarch64-buildroot-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER
        /home/IM/aarch64_toolchain/aarch64_gun/bin/aarch64-buildroot-linux-gnu-g++)
    set(CMAKE_FIND_ROOT_PATH
        /home/IM/aarch64_toolchain/aarch64_gun/aarch64-buildroot-linux-gnu/sysroot)
    ```
- 在**cmake/third_library.cmake**中配置arm 版本cuda与tensorrt交叉编译环境
    ```shell
    # Setting up your computer's cuda and tensorrt environment -- arm version
    set(CUDA_TOOLKIT_ROOT_DIR "/home/aarch64_toolchain/cuda")
    set(TENSORRT_DIR "/home/aarch64_toolchain/tensorrt")
    ```
### 1.2 Configuring third-party library paths in a Docker environment
- Configure the third-party library path in **cmake/third_library.cmake**, where ${COMPILER_FLAG} is the name of the customized directory for storing third-party libraries for the arm and x86 versions, e.g., the repository provides the following: aarch64_toolchain (arm version), x86_toolchain (x86 version).

    The author unified the third-party libraries in a fixed directory on the local computer, and mounted the third-party library directories into the container when building docker containers (e.g., put all the directories in Figure 1.2 under the local home path, and mount the home directory into the container when building docker containers), such as in Figure 1.2.
    ![library](./image.png)
    ![aarch64_toolchain](./aarch64_toolchain.png)
    ```shell
    set(GLOG_DIR "/home/IM/${COMPILER_FLAG}/glog0.6.0")
    set(EIGIN_DIR "/home/IM/${COMPILER_FLAG}/eigen3.4")
    set(GFLAGS_DIR "/home/IM/${COMPILER_FLAG}/gflags_2.2.2")
    set(OPENCV_DIR "/home/IM/${COMPILER_FLAG}/opencv3.4.5")
    set(YAMLCPP_DIR "/home/IM/${COMPILER_FLAG}/yaml_cpp")
    set(TENSORRT_DIR "/home/IM/${COMPILER_FLAG}/tensorrt")
    ```
### 1.3 Source Compilation
![project](https://img.shields.io/badge/How%20to%20build-project-brightgreen)

Shell scripts provide optional platforms.

- Available platform parameters are as follows:
    |NVIDIA|QNN|
    |:-:|:-:|
- Available model parameters are as follows:
    |yolov5|yolox|
    |:-:|:-:|

> **Note:**
>
> - Available parameters are as follows:
>   1) **-a | -all | all**. Compile all modules
>   2) **-clean**. Clear compiled files
>   3) **-arm**.   Enable cross-compilation mode
>   4) **-debug**. Enable debug mode
>   5) **-x86**.   Enable x86 mode
>   6) **-pack**.  Packaging of executables and dynamic libraries

```shell
#!/bin/bash
- x86
bash ./scripts/build.sh yolov5 nvidia  -x86 -pack -clean

- arm
bash ./scripts/build.sh yolov5 nvidia  -arm -pack -clean
```

### 1.4 program execution
![Usage](https://img.shields.io/badge/How%20to%20use-platform-brightgreen)

```shell
#!/bin/bash
- x86
bash ./install_nvidia/run.sh yolov5 nvidia

- arm
bash ./install_nvidia/run.sh yolov5 nvidia -arm
```
