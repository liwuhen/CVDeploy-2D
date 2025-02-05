/* ==================================================================
* Copyright (c) 2024, LiWuHen.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an
 BASIS
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ===================================================================
*/

#ifndef APP_COMMON_ENUM_MSG_H__
#define APP_COMMON_ENUM_MSG_H__

#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace hpc {
namespace common {

typedef enum class model_acc_ : uint8_t { MODEL_FLOAT32 = 0, MODEL_FLOAT16, MODEL_INT8 } ModelACC;

typedef enum class device_mode_ : uint8_t { GPU_MODE = 0, CPU_MODE } DeviceMode;

typedef enum class app_yolo_ : uint8_t { YOLOV5_MODE = 0, YOLOX_MODE } AppYolo;

typedef enum class yolo_decode_ : uint8_t { FEATURE_LEVEL = 0, INPUT_LEVEL } DecodeType;

typedef enum class yolo_decode_branch_ : uint8_t {
    FEATURE_ONE = 0,
    FEATURE_SECOND,
    FEATURE_THREE
} DecodeBranch;

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_ENUM_MSG_H__
