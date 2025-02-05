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

#ifndef APP_COMMON_COLORPRINTF_H__
#define APP_COMMON_COLORPRINTF_H__

#include <glog/logging.h>
#include <stdio.h>

namespace hpc {
namespace common {

// 颜色宏定义
#define NONEC "\033[0m"
#define RED "\033[1;32;31m"
#define LIGHT_RED "\033[1;31m"
#define GREEN "\033[1;32;32m"
#define LIGHT_GREEN "\033[1;32m"
#define BLUE "\033[1;32;34m"
#define LIGHT_BLUE "\033[1;34m"
#define DARY_GRAY "\033[1;30m"
#define CYAN "\033[1;36m"
#define LIGHT_CYAN "\033[1;36m"
#define PURPLE "\033[1;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN "\033[1;33m"
#define YELLOW "\033[1;33m"
#define LIGHT_GRAY "\033[1;37m"
#define WHITE "\033[1;37m"

#define LOG_COLOR(severity, message) LOG(severity) << message
#define GLOG_ERROR(message) LOG_COLOR(ERROR, message)
#define GLOG_WARNING(message) LOG_COLOR(WARNING, message)
#define GLOG_INFO(message) LOG_COLOR(INFO, message)

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_COLORPRINTF_H__
