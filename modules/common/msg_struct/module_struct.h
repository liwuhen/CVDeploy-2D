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

#ifndef APP_COMMON_MODULE_STRUCT_H__
#define APP_COMMON_MODULE_STRUCT_H__

#include <iostream>

#include "common.hpp"

namespace hpc {

namespace common {

using namespace std;

typedef enum module_type : uint8_t {
  MODULE_INPUT = 0,
  MODULE_INFERENCE,
  MODULE_POSTPROCESS,
  MODULE_BUT
} ModuleType;

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_MODULE_STRUCT_H__
