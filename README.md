# AI model deployment based on NVIDIA and Qualcomm platforms

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white)
![Parallel Computing](https://img.shields.io/badge/Parallel-Computing-orange?style=for-the-badge)
![HPC](https://img.shields.io/badge/HPC-High%20Performance%20Computing-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMiAxN3YtMmgtM3YtM2gydi0yaDJ2LTJoLTR2N2gtN3YtN0g4djhoLTNWM0gzdjE4aDE4di00eiIvPjwvc3ZnPg==)
![Performance](https://img.shields.io/badge/Performance-Optimized-red?style=for-the-badge)
![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

The repository mainly provides model inference functionality, and the code provides daily development of packaged libs for integration, testing, and inference. The framework provides multi-threaded, singleton pattern, producer and consumer patterns. Cache log analysis is also supported. At the same time, this repository also supports autopilot driving, parking, and cockpit areas.

# ![third-party](https://img.shields.io/badge/third-party-blue) Third-party Libraries

|Libraries|Eigen|Gflags|Glog|Yaml-cpp|Cuda|Cudnn|Tensorrt|Opencv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Version|3.4|2.2.2|0.6.0|0.8.0|11.4|8.4|8.4|3.4.5|

# ![docker](https://img.shields.io/badge/How%20to%20build-docker-brightgreen) Cross-compile Docker

- **Docker Image**
```shell
#!/bin/bash

./docker/build_docker.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag cross-aarch64_dev
```

- **Docker Container**
```shell
#!/bin/bash

docker run -it --gpus all --network host --privileged  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/:/home/  --name cross-aarch64_container cross-aarch64_dev:latest  bash
```

# ![project](https://img.shields.io/badge/How%20to%20build-project-brightgreen) Building Project
## 1. Setup Environment
###  Shell Example:  X86
```shell 
# Setting up your computer's cuda and tensorrt environment -- x86 version
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4/targets/x86_64-linux")
set(TENSORRT_DIR "/home/x86_toolchain/tensorrt")
```
###  Shell Example:  Arm
- **arm_toolchain.cmake**
```shell
# Setting up your computer's arm gcc,g++ environment -- arm version
set(CMAKE_C_COMPILER  /home/aarch64_toolchain/aarch64_gun/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /home/aarch64_toolchain/aarch64_gun/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH  /home/aarch64_toolchain/aarch64_gun/aarch64-buildroot-linux-gnu/sysroot)
```
- **thirdlibs.cmake**
```shell
# Setting up your computer's cuda and tensorrt environment -- arm version
set(CUDA_TOOLKIT_ROOT_DIR "/home/aarch64_toolchain/cuda")
set(TENSORRT_DIR "/home/aarch64_toolchain/tensorrt")
```

## 2. X86 && Arm Platform
Shell scripts provide optional platforms.

- Available platform parameters are as follows:
    |NVIDIA|QNN| 
    |:-:|:-:|
- Available model parameters are as follows: 
    |yolov5|yolox| 
    |:-:|:-:|
- Available parameters are as follows: 
    1) **-a | -all | all**. Compile all modules
    2) **-clean**. Clear compiled files
    3) **-arm**.   Enable cross-compilation mode
    4) **-debug**. Enable debug mode
    5) **-x86**.   Enable x86 mode
    6) **-pack**.  Packaging of executables and dynamic libraries


```shell
#!/bin/bash
- x86
bash ./scripts/build.sh yolov5 nvidia  -x86 -pack -clean

- arm
bash ./scripts/build.sh yolov5 nvidia  -arm -pack -clean
```

# ![Usage](https://img.shields.io/badge/How%20to%20use-platform-brightgreen) Usage X86 && ARM Linux

```shell
#!/bin/bash
- x86 
bash ./install_nvidia/run.sh yolov5 nvidia 

- arm
bash ./install_nvidia/run.sh yolov5 nvidia -arm
```

# ![Contribute](https://img.shields.io/badge/how%20to%20contribute-project-brightgreen) Contributing
Welcome users to participate in these projects. Please refer to [CONTRIBUTING.md](./doc/CONTRIBUTING.md) for the contributing guideline.We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in Working Groups, Working Groups have most of their discussions on [Slack](https://app.slack.com/client/T07U5CEEXCP/C07UKUA9TCJ) .


# References
- [Yolov5](https://github.com/ultralytics/yolov5)
- [Yolox](https://github.com/Megvii-BaseDetection/YOLOX)

- **Blog**：[Setup Environment](https://zhuanlan.zhihu.com/p/818205320)