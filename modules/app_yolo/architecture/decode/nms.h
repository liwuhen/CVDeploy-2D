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

#ifndef APP_YOLO_NMS_H__
#define APP_YOLO_NMS_H__

#include <glog/logging.h>
#include <math.h>
#include <stdio.h>

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "iou.h"
#include "logger.h"
#include "parseconfig.h"
#include "plugin.h"
#include "task_struct.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace hpc::common;

/**
 * @class NmsPlugin.
 * @brief Nms function.
 */
class NmsPlugin : public PluginNodeBase {
 public:
  NmsPlugin();
  ~NmsPlugin();

  /**
   * @brief     init．
   * @param[in] void．
   * @return    bool.
   */
  bool Init() override;

  /**
   * @brief     Configuration parameters.
   * @param[in] shared_ptr<ParseMsgs>&.
   * @return    bool.
   */
  bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) override;

  /**
   * @brief     Nms.
   * @param[in] vector<Box>& boxes, vector<Box>& box_result, float nms_threshold.
   * @return    bool.
   */
  bool Nms(vector<Box>& boxes, vector<Box>& box_result, float nms_threshold);

 private:
  shared_ptr<IouPlugin> iou_plugin_;
  std::shared_ptr<ParseMsgs> parsemsgs_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_NMS_H__
