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

set -e

# common param
TIME=$(date "+%Y-%m-%d %H:%M:%S")
export HOME_DIR=$(dirname "$(readlink -f "$0")")
HOME_DIR="${HOME_DIR}"/..
echo -e "\e[1m\e[34m[Bash-Home-${TIME}]: ${HOME_DIR} \e[0m"


# compiler configure
CMAKE_COMPILER_PATH="${HOME_DIR}"/platforms/linux/arm-toolchain.cmake
CMAKE_ARM_ARGS="-DCMAKE_TOOLCHAIN_FILE=${CMAKE_COMPILER_PATH}"

# configure  param
PACK_FLAG=OFF
CLEAN_FLAG=OFF
PC_X86_FLAG=OFF
CROSS_COMPILE=OFF
ALL_MODEL_FLAG=OFF
FUN_BOOL_FLAG=True
TBUILD_VERSION=Release
CONFIGURE_SETS=("all" "clean" "arm" "x86" "pack")

# model param
MODEL_FLAG=NONE
MODEL_BOOL_FLAG=False
MODEL_SETS=("yolov5" "yolov8" "yolov11" "yolox")

# compiler platform
PLATFORM_FLAG=NONE
PLATFORM_BOOL_FLAG=False
PLATFORM_SETS=("NVIDIA" "QNN")

function parse_args()
{
    for opt in "$@" ; do
        # model
        for model in "${MODEL_SETS[@]}"; do
            if [ "${opt,,}" == "$model" ]; then
                MODEL_FLAG=${opt,,}
                echo -e "\e[1m\e[34m[Bash-Model-${TIME}]: $opt in model sets \e[0m"
                MODEL_BOOL_FLAG=True
                break
            fi
        done

        # platform
        for platform in "${PLATFORM_SETS[@]}"; do
            if [ "${opt^^}" == "$platform" ]; then
                PLATFORM_FLAG=$platform
                echo -e "\e[1m\e[34m[Bash-Platform-${TIME}]: $opt in platform sets \e[0m"
                PLATFORM_BOOL_FLAG=True
                break
            fi
        done

        # function
        if [[ "$opt" == -* ]]; then
            case "$opt" in
                -all)    ALL_MODEL_FLAG=ON    ;;
                -clean)  CLEAN_FLAG=ON        ;;
                -arm)    CROSS_COMPILE=ON     ;;
                -x86)    PC_X86_FLAG=ON       ;;
                -pack)   PACK_FLAG=ON         ;;
                *)       FUN_BOOL_FLAG=False  ;;
            esac
        else
            for fun in "${CONFIGURE_SETS[@]}"; do
                if [ "$opt" == "$fun" ]; then
                    FUN_BOOL_FLAG=False
                    break
                fi
            done
        fi

    done

    # check model
    if [ "$MODEL_BOOL_FLAG" == "False" ] ; then
            echo -e "\e[1m\e[34m[Bash-Model-${TIME}]: parameters not in model sets.
Available model parameters are as follows:
    1) yolov5    2) yolov8    3) yolov11    4) yolox \e[0m"
            exit 1
    fi
    # check platform
    if [ "$PLATFORM_BOOL_FLAG" == "False" ] ; then
        echo -e "\e[1m\e[34m[Bash-Model-${TIME}]: parameters not in platform sets.
Available platform parameters are as follows:
    1) NVIDIA    2) QNN \e[0m"
        exit 1
    fi
    # check function
    if [ "$FUN_BOOL_FLAG" == "False" ] ; then
            echo -e "\e[1m\e[34m[Bash-Model-${TIME}]: parameters does not exist.
Available parameters are as follows:
    1) -a | -all | all. Compile all modules
    2) -clean. Clear compiled files
    3) -arm.   Enable cross-compilation mode
    4) -x86.   Enable x86 mode
    5) -pack.  Packaging of executables and dynamic libraries\e[0m"
        exit 1
    fi

}
