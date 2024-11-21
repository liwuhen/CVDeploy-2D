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

#ifndef APP_YOLO_POSTPROCESSOR_H__
#define APP_YOLO_POSTPROCESSOR_H__

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

#include "common.hpp"
#include "inference.h"
#include "module.h"
#include "parseconfig.h"
#include "std_buffer.h"
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

/**
 * @class PostProcessor.
 * @brief Algorithmic post-processing.
 */
class PostProcessor : public InferModuleBase {
 public:
  PostProcessor();
  ~PostProcessor();

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

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease();

  /**
   * @brief     Inference.
   * @param[in] void.
   * @return    bool.
   */
  void Inference();

  /**
   * @brief     Visualisation.
   * @param[in] [bool, cv::Mat&, int64_t, vector<Box>&].
   * @return    void.
   */
  void Visualisation(bool real_time, cv::Mat& img, int64_t timestamp, vector<Box>& results);

 private:
  std::atomic<bool> running_;

  InfertMsg output_msg_;

  shared_ptr<ParseMsgs> parsemsgs_;
  std::shared_ptr<std::thread> worker_ptr_;
  std::shared_ptr<InferenceEngine> inference_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_POSTPROCESSOR_H__
