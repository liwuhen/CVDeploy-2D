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

#ifndef APP_YOLO_PLUGIN_H__
#define APP_YOLO_PLUGIN_H__

#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "module_struct.h"
#include "parseconfig.h"
/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace hpc::common;
/**
 * @class PluginNodeBase.
 * @brief Plugin node base.
 */
class PluginNodeBase {
 public:
  PluginNodeBase() {}
  virtual ~PluginNodeBase() = default;

  virtual bool Init() = 0;

  virtual bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) = 0;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_PLUGIN_H__
