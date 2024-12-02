# Installation
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white)

## 1. 环境搭建
### 1.1 docker开发交叉编译环境搭建
本仓库提供两种docker环境的搭建方式，第一种直接下载docker镜像即可开发的完整镜像，第二种分离第三方库存储在本地环境，下载最小化的镜像。
- 第一种完整镜像

    镜像地址：Docker hub地址：[cross-aarch64_dev](https://hub.docker.com/r/liwuhen/cross-aarch64_dev)

    容器启动命令：
    ```shell
    docker run -it --gpus all --network host --privileged  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/:/home/  --name cross-aarch64_container cross-aarch64_dev:latest  bash
    ```

- 第二种最小化镜像（推荐）

    Docker file生成镜像：
    ```shell
    #!/bin/bash
    ./docker/build_docker.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag cross-aarch64_dev
    ```
    容器启动命令：
    ```shell
    docker run -it --gpus all --network host --privileged  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/:/home/  --name cross-aarch64_container cross-aarch64_dev:latest  bash
    ```

### 1.2 本地交叉编译环境搭建

    配置第三方库环境目录，作者将第三方库统一存放在本地电脑的固定目录，配置环境变量，方便交叉编译。

![library1](./image.png)
![aarch64_toolchain2](./aarch64_toolchain.png)
