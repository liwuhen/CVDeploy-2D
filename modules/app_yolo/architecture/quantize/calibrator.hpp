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

#ifndef APP_YOLO_QUANTIZE_CALIBRATOR_H__
#define APP_YOLO_QUANTIZE_CALIBRATOR_H__

#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include "class_factory.h"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace nvinfer1;

typedef std::function<void(
    int current, int count, float* ptensor,
    const std::vector<std::string>& files,
    std::shared_ptr<ParseMsgs>& parsemsgs
)> Int8Process;

/**
 * @description: int8 熵校准器：用于评估量化前后的分布改变.
 */
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
  Int8EntropyCalibrator() {}

  virtual ~Int8EntropyCalibrator() {}

  void Init(const vector<string>& imagefiles,
      nvinfer1::Dims dims, const Int8Process& preprocess,
      std::shared_ptr<ParseMsgs>& parsemsgs) {
    assert(preprocess != nullptr);
    this->dims_       = dims;
    this->allimgs_    = imagefiles;
    this->parsemsgs_  = parsemsgs;
    this->preprocess_ = preprocess;
    this->fromCalibratorData_ = false;
    files_.resize(dims.d[0]);
  }

  // 这个构造函数，是允许从缓存数据中加载标定结果，这样不用重新读取图像处理
  void Init(const vector<uint8_t>& entropyCalibratorData,
      nvinfer1::Dims dims, const Int8Process& preprocess,
      std::shared_ptr<ParseMsgs>& parsemsgs) {
    assert(preprocess != nullptr);
    this->dims_       = dims;
    this->parsemsgs_  = parsemsgs;
    this->preprocess_ = preprocess;
    this->fromCalibratorData_ = true;
    this->entropyCalibratorData_ = entropyCalibratorData;
    files_.resize(dims.d[0]);
  }

  // 想要按照多少的 batch 进行标定
  int getBatchSize() const noexcept {
    return dims_.d[0];
  }

  bool next() {
    int batch_size = dims_.d[0];
    if ( cursor_ + batch_size > allimgs_.size() )
      return false;

    for ( int i = 0; i < batch_size; ++i )
      files_[i] = allimgs_[cursor_++];

    if( tensor_host_ == nullptr ) {
      size_t volumn = 1;
      for( int i = 0; i < dims_.nbDims; ++i )
        volumn *= dims_.d[i];

      bytes_ = volumn * sizeof(float);
      checkRuntime(cudaMallocHost(&tensor_host_, bytes_));
      checkRuntime(cudaMalloc(&tensor_device_, bytes_));
    }

    preprocess_(cursor_, allimgs_.size(), tensor_host_, files_, parsemsgs_);
    checkRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));
    return true;
  }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (!next()) return false;
    bindings[0] = tensor_device_;
    return true;
  }

  const vector<uint8_t>& getEntropyCalibratorData() {
    return entropyCalibratorData_;
  }

  const void* readCalibrationCache(size_t& length) noexcept {
    if (fromCalibratorData_) {
      length = this->entropyCalibratorData_.size();
      return this->entropyCalibratorData_.data();
    }

    length = 0;
    return nullptr;
  }

  virtual void writeCalibrationCache(const void* cache, size_t length) noexcept {
    // entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
    ofstream output(parsemsgs_->calib_table_path_, ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
  }

  bool MemFree() {
    if ( tensor_host_ != nullptr ) {
      checkRuntime(cudaFreeHost(tensor_host_));
      checkRuntime(cudaFree(tensor_device_));
      tensor_host_   = nullptr;
      tensor_device_ = nullptr;
    }
  }

private:
  Int8Process preprocess_;
  vector<string> allimgs_;
  size_t batch_size_ = 0;
  int cursor_   = 0;
  size_t bytes_ = 0;
  nvinfer1::Dims dims_;
  vector<string> files_;
  float* tensor_host_   = nullptr;
  float* tensor_device_ = nullptr;
  vector<uint8_t> entropyCalibratorData_;
  bool fromCalibratorData_ = false;
  std::shared_ptr<ParseMsgs> parsemsgs_;
};

REGISTER_CLASS("Int8EntropyCalibrator", Int8EntropyCalibrator);

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_TRT_INFER_H__
