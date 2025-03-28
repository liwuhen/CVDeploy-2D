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

#ifndef APP_YOLO_INFERENCE_H__
#define APP_YOLO_INFERENCE_H__

#include <glog/logging.h>

#include <iostream>

#include "common.hpp"
#include "decodeprocessor.h"
#include "logger.h"
#include "module.h"
#include "parseconfig.h"
#include "preprocessor.h"
#include "std_time.h"
#include "task_struct.hpp"
#include "trt_infer.h"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace cv;
using namespace std;
using namespace nvinfer1;
using namespace hpc::common;

/**
 * @class InferenceEngine.
 * @brief Model inference.
 */
class InferenceEngine : public InferModuleBase {
 public:
  InferenceEngine();
  ~InferenceEngine();

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
   * @brief     Load data.
   * @param[in] InfertMsg&.
   * @return    bool.
   */
  bool LoadData(InfertMsg& infer_msg);

  /**
   * @brief     TriggerCallback.
   * @param[in] vector<InfertMsg>&
   * @return    void.
   */
  void TriggerCallback(vector<InfertMsg>& infer_msg_vec);

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease();
  /**
   * @brief     Inference.
   * @param[in] void．
   * @return    bool.
   */
  bool Inference();
  /**
   * @brief     Python model output binary for post-processing inference.
   * @param[in] void．
   * @return    bool.
   */
  bool InferenceV1();
  /**
   * @brief     Timer.
   * @param[in] [int, std::function<void()>]
   * @return    void.
   */
  void TimeManager(int interval, function<void()> task);

 public:
  std::shared_ptr<InferMsgQue> bboxQueue_;
  std::vector<InfertMsg> callbackMsg;

 private:
  std::atomic<bool> running_;

  float* output_img_device_;

  InfertMsg input_msg_;
  shared_ptr<InputMsgQue> inputMsgQue_;

  shared_ptr<ParseMsgs> parsemsgs_;
  shared_ptr<TrtInfer> trtInfer_;
  shared_ptr<PreProcessor> preProcessor_;
  shared_ptr<DecodeProcessor> decodeProcessor_;

  std::shared_ptr<std::thread> worker_ptr_;

  std::vector<double>preproccess_time_;
  std::vector<double>infer_time_;
  std::vector<double>decode_time_;
  std::vector<double>endtoend_time_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_INFERENCE_H__
