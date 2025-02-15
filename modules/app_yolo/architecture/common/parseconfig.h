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

#ifndef APP_YOLO_PARSECONFIG_H__
#define APP_YOLO_PARSECONFIG_H__

#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "appconfig.h"
#include "colorprintf.h"

/**
 * @namespace hpc::common
 * @brief hpc::common
 */
namespace hpc {
namespace common {

using namespace std;

/**
 * @class ParseMsgs.
 * @brief Parsing configuration file.
 */
class ParseMsgs {
 public:
  ParseMsgs();
  ~ParseMsgs();

  /**
   * @brief     Read yaml file.
   * @param[in] void．
   * @return    void.
   */
  void ReadYamlParam();

 public:
  bool is_init_;
  int src_img_w_;        // Raw image width
  int src_img_h_;        // Raw image height
  int src_img_c_;        // Raw image channel
  int srcimg_size_;      // Raw image size
  int dst_img_w_;        // Target image width
  int dst_img_h_;        // Target image height
  int dst_img_c_;        // Target image channel
  int dstimg_size_;      // Target image size
  int model_acc_;        // Model quantisation accuracy
  int branch_num_;       // Model branch number
  int batchsizes_;       // Batch size
  int infer_mode_;       // infer mode, 0-gpu, 1-cpu, 2-npu
  int decode_type_;      // Decode type
  int max_objects_;      // Maximum number of targets
  int input_msgdepth_;   // Input msg queue length
  int decode_msgdepth_;  // Decode msg queue length
  float obj_threshold_;  // Target Thresholds
  float nms_threshold_;  // Nms Target Thresholds

  std::string img_path_;     // Offline original image path
  std::string save_img_;     // Save result image paths offline
  std::string yaml_path_;    // Yaml path
  std::string trt_path_;     // trt file path
  std::string onnx_path_;    // onnx file path
  std::string predict_path_; // Predictions of onnx model python outputs
  std::string log_path_;     // log file path
  std::string imgs_path_;    // Offline test image collection path

  std::vector<std::vector<int>> predict_dim_;  // Model prediction output dimensions
  std::vector<std::vector<int>> branchs_dim_;
};

}  // namespace common
}  // namespace hpc

#endif  // APP_YOLO_PARSECONFIG_H__
