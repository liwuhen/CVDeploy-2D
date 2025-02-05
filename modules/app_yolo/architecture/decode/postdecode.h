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

#ifndef APP_YOLO_POSTDECODE_H__
#define APP_YOLO_POSTDECODE_H__

#include <glog/logging.h>
#include <stdio.h>
#include <time.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "nms.h"
#include "common.hpp"
#include "dataset.h"
#include "module.h"
#include "decode.h"
#include "parseconfig.h"
#include "task_struct.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace cv;
using namespace hpc::common;

using AnchorPointsVector = std::vector<std::vector<std::pair<int, int>>>;using AnchorPointsVector = std::vector<std::vector<std::pair<int, int>>>;

/**
 * @class ModelDecode.
 * @brief Bbox decode.
 */
class ModelDecode : public DecodeModuleBase {
 public:
  ModelDecode() {}
  ~ModelDecode() {}

  /**
   * @brief     init．
   * @param[in] void．
   * @return    bool.
   */
  bool Init() override;

  /**
   * @brief     The inference algorithm handles threads．
   * @param[in] void．
   * @return    bool.
   */
  bool RunStart() override;

  /**
   * @brief     Thread stop．
   * @param[in] void．
   * @return    bool.
   */
  bool RunStop() override;

  /**
   * @brief     Software function stops．
   * @param[in] void．
   * @return    bool.
   */
  bool RunRelease() override;

  /**
   * @brief     Configuration parameters.
   * @param[in] shared_ptr<ParseMsgs>&.
   * @return    bool.
   */
  bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) override;

  /**
   * @brief     Cal anchor．
   * @param[in] ．
   * @return    AnchorPointsVector.
   */
  AnchorPointsVector Generate_Anchor_Points();
  /**
   * @brief     Box decode feature level．
   * @param[in] [float*, InfertMsg&, vector<Box>&]．
   * @return    void.
   */
  void BboxDecodeFeatureLevel(std::vector<float*>& predict,
    InfertMsg& infer_msg, vector<Box>& box_result);

  /**
   * @brief     Box decode input level．
   * @param[in] [float*, InfertMsg&, vector<Box>&]．
   * @return    void.
   */
  void BboxDecodeInputLevel(std::vector<float*>& predict,
    InfertMsg& infer_msg, vector<Box>& box_result);

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease() {};

  /**
   * @brief     Bbox mapping to original map scale.
   * @param[in] [vector<Box>&, std::map<string, pair<int, int>>&]．
   * @return    void.
   */
  void ScaleBoxes(vector<Box>& box_result);

 private:
  std::atomic<bool> running_;

  InfertMsg output_msg_;

  AnchorPointsVector anchor_points_;

  shared_ptr<NmsPlugin> nms_plugin_;

  shared_ptr<ParseMsgs> parsemsgs_;

  std::map<string, pair<int, int>> imgshape_;
  std::vector<std::pair<int, int>> feat_sizes_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_POSTDECODE_H__
