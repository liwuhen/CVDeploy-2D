#!/usr/bin/env bash
#
# ==================================================================
# Copyright (c) 2024, LiWuHen.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an
# BASIS
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================
#/

ARG CUDA_VERSION=11.4.3
ARG OS_VERSION=18.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.4.3.1
ENV DEBIAN_FRONTEND=noninteractive

# Install required libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    g++-9 \
    gcc-9 \
    zlib1g-dev \
    libc++-dev \
    libgtk2.0-dev\
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libncurses5-dev \
    libgdbm-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    build-essential

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install Python3.9
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz && \
    tar -xzf Python-3.9.16.tgz && \
    cd Python-3.9.16 && \
    ./configure --enable-optimizations && \
    make -j2 && \
    make altinstall && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.9 1 && \
    python --version

RUN cd /usr/local/bin &&\
    ln -s /usr/local/bin/python3.9 python &&\
    ln -s /usr/local/bin/pip3.9 pip
RUN pip3.9 install --upgrade pip
RUN pip3.9 install setuptools>=41.0.0

# USER trtuser
RUN ["/bin/bash"]
