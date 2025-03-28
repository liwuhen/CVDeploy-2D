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

#ifndef APP_YOLO_PREPROCESSOR_H__
#define APP_YOLO_PREPROCESSOR_H__

#include <glog/logging.h>
#include <time.h>

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

#include "common.hpp"
#include "enum_msg.h"
#include "module.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "parseconfig.h"
#include "std_cmake.h"
#include "task_struct.hpp"
#include "utils.hpp"
#include "warpaffine.hpp"
#include "preprocess_registry.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace Eigen;
using namespace hpc::common;

/**
 * @class PreProcessor.
 * @brief Model preprocessing.
 */
class PreProcessor : public InferModuleBase {
 public:
  PreProcessor();
  ~PreProcessor();

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
   * @brief     Cpu and gpu memory free.
   * @param[in] void.
   * @return    bool.
   */
  bool MemFree();

  /**
   * @brief     Inference.
   * @param[in] float*.
   * @return    bool.
   */
  bool Inference(InfertMsg& input_msg, float* dstimg, DeviceMode inferMode, cudaStream_t stream);

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease();

  /**
   * @brief     Memory allocator.
   * @param[in] void．
   * @return    bool.
   */
  bool MemAllocator();

  /**
   * @brief     cal affine matrix.
   * @param[in] InfertMsg&
   * @return    void.
   */
  void CalAffineMatrix(InfertMsg& input_msg);

  /**
   * @brief     Gpu preprocessor.
   * @param[in] [InfertMsg&, int, float *, cudaStream_t].
   * @return    bool.
   */
  bool GpuPreprocessor(InfertMsg& input_msg,
      float* dstimg, cudaStream_t stream);

  /**
   * @brief     Cpu preprocessor.
   * @param[in] [cv::Mat&, float *, cudaStream_t].
   * @return    bool.
   */
  bool CpuPreprocessor(InfertMsg& input_msg,
      float* dstimg, cudaStream_t stream);

  /**
   * @brief     Load file.
   * @param[in] [const string&].
   * @return    bool.
   */
  std::vector<uint8_t>LoadFile(const string& file);

 private:
  float* input_data_host_;
  uint8_t* input_data_device_;

  std::shared_ptr<ParseMsgs> parsemsgs_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_PREPROCESSOR_H__
