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

#ifndef APP_COMMON_STD_TIME_H__
#define APP_COMMON_STD_TIME_H__

#include <iostream>
#include <mutex>
#include <memory>
#include <queue>
#include <string>
#include <condition_variable>
#include <list>
#include <chrono>
#include <functional>

namespace hpc {
namespace common {

static time_t GetSystmeTime()
{
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::system_clock::now().time_since_epoch()).count());
}


} //namespace common
} //namespace hpc

#endif // APP_COMMON_STD_TIME_H__