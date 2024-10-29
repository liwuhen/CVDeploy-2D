#!/bin/bash
set -e  # 开启错误检测

. ./scripts/bash_config.sh

parse_args "$@"
INSTALL_PATH=${HOME_DIR}/install_${PLATFORM_FLAG,,}

if [ $INSTALL_PATH != "" ] ; then 

    if [ "$CROSS_COMPILE" == "OFF" ]; then
        INSTALL_PATH=$INSTALL_PATH/${MODEL_FLAG,,}_bin/x86
    else
        INSTALL_PATH=$INSTALL_PATH/${MODEL_FLAG,,}_bin/arm
    fi
    echo -e "\e[1m\e[34m[Bash-Run-${TIME}]: $INSTALL_PATH  \e[0m"

fi
TENSORRT=/home/x86_toolchain/tensorrt
THIRD_PARTY=$INSTALL_PATH/third_party
export LD_LIBRARY_PATH=$INSTALL_PATH/proprecessor:$INSTALL_PATH/libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$THIRD_PARTY/opencv:$TENSORRT/lib:$THIRD_PARTY/zbar:$LD_LIBRARY_PATH
cd $INSTALL_PATH && ./${MODEL_FLAG,,}_project_exe