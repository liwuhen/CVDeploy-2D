#!/bin/bash
set -e

# common param
TIME=$(date "+%Y-%m-%d %H:%M:%S")
export HOME_DIR=$(dirname $(readlink -f $0))
HOME_DIR=${HOME_DIR}/..
echo -e "\e[1m\e[34m[Bash-Home-${TIME}]: ${HOME_DIR} \e[0m"


# compiler configure
COMPLIER_FLAG="x86"
CMAKE_COMPILER_PATH=${HOME_DIR}/platforms/linux/arm-toolchain.cmake
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
MODEL_SETS=("yolov5" "yolox")

# complier platform
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
    1) yolov5    2) yolox \e[0m"
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