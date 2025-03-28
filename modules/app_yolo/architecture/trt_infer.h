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

#ifndef APP_YOLO_TRT_INFER_H__
#define APP_YOLO_TRT_INFER_H__

#include <glog/logging.h>
#include <math.h>
#include <stdio.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "common.hpp"
#include "enum_msg.h"
#include "logger.h"
#include "module.h"
#include "parseconfig.h"
#include "std_buffer.h"
#include "std_cmake.h"
#include "task_struct.hpp"
#include "utils.hpp"
#include "file.hpp"
#include "calibrator.hpp"
#include "function_registry.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace hpc::common;

/**
 * @class TrtInfer.
 * @brief Trt model infer.
 */
class TrtInfer : public InferModuleBase {
 public:
  TrtInfer();
  ~TrtInfer();

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
   * @param[in] void．
   * @return    bool.
   */
  bool MemFree();

  /**
   * @brief     Load file.
   * @param[in] const string&.
   * @return    bool.
   */
  std::vector<uint8_t> LoadFile(const string& file);

  /**
   * @brief     Load image file.
   * @param[in] std::vector<string>&.
   * @return    bool.
   */
  void LoadCalibDataFile(const std::string& path, std::vector<string>& data);

  /**
   * @brief     Inference.
   * @param[in] float*.
   * @return    bool.
   */
  bool Inference(float* output_img_device);

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease();

  /**
   * @brief     Build trt model from onnx.
   * @param[in] void．
   * @return    bool.
   */
  bool BuildModel();

  /**
   * @brief     Parse model.
   * @param[in] void．
   * @return    bool.
   */
  bool ParseModel();

  /**
   * @brief     Generate smart pointer for nvidia function.
   * @param[in] _T．
   * @return    _T.
   */
  template <typename _T>
  shared_ptr<_T> make_nvshared(_T* ptr) {
    return shared_ptr<_T>(ptr, [](_T* p) { p->destroy(); });
  }

  /**
   * @brief     Memory allocator.
   * @param[in] void．
   * @return    bool.
   */
  bool MemAllocator();

 public:
  std::vector<float *> cpu_buffers_;

 private:
  bool hasEntropyCalibrator_;

  std::vector<float *> gpu_buffers_;

  cudaStream_t stream_;
  std::map<std::string, int> in_out_size_;
  std::map<std::string, std::vector<std::string>> binding_names_;
  std::map<std::string, std::pair<int, size_t>> engine_name_size_;

  std::shared_ptr<IExecutionContext> execution_context_;

  std::shared_ptr<ParseMsgs> parsemsgs_;
  std::shared_ptr<Int8EntropyCalibrator> calib_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_TRT_INFER_H__
