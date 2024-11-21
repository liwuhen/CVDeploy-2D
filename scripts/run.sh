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

. ./scripts/config.sh

parse_args "$@"
INSTALL_PATH="${HOME_DIR}"/install_"${PLATFORM_FLAG,,}"

if [ "${INSTALL_PATH}" != "" ] ; then

    if [ "${CROSS_COMPILE}" == "OFF" ]; then
        INSTALL_PATH="${INSTALL_PATH}"/"${MODEL_FLAG,,}"_bin/x86
    else
        INSTALL_PATH="${INSTALL_PATH}"/"${MODEL_FLAG,,}"_bin/arm
    fi
    echo -e "\e[1m\e[34m[Bash-Run-${TIME}]: ${INSTALL_PATH}  \e[0m"

fi
TENSORRT=/home/IM/x86_toolchain/tensorrt
THIRD_PARTY="${INSTALL_PATH}"/third_party
export LD_LIBRARY_PATH=$INSTALL_PATH/proprecessor:$INSTALL_PATH/libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$THIRD_PARTY/opencv:$TENSORRT/lib:$THIRD_PARTY/zbar:$LD_LIBRARY_PATH
cd "${INSTALL_PATH}" && ./"${MODEL_FLAG,,}"_project_exe
