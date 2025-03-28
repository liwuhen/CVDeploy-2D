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

#include "preprocessor.h"

namespace hpc {

namespace appinfer {

PreProcessor::PreProcessor() {}

PreProcessor::~PreProcessor() {}

/**
 * @description: init．
 */
bool PreProcessor::Init() {
  MemAllocator();

  GLOG_INFO("[Init]: PreProcessor module init ");
  return true;
}

/**
 * @description: The inference algorithm handles threads．
 */
bool PreProcessor::RunStart() {
  GLOG_INFO("[RunStart]: PreProcessor module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool PreProcessor::RunStop() {
  GLOG_INFO("[RunStop]: PreProcessor module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool PreProcessor::RunRelease() {
  GLOG_INFO("[RunRelease]: PreProcessor module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool PreProcessor::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: PreProcessor module set param failed ");
    return false;
  }

  GLOG_INFO("[SetParam]: PreProcessor module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool PreProcessor::DataResourceRelease() {}

/**
 * @description: Inference.
 */
bool PreProcessor::Inference(InfertMsg& input_msg,
    float* dstimg, DeviceMode inferMode, cudaStream_t stream) {

  CalAffineMatrix(input_msg);

  switch (inferMode) {
    case DeviceMode::GPU_MODE:
      if (!GpuPreprocessor(input_msg, dstimg, stream)) {
        return false;
      }
      break;
    case DeviceMode::CPU_MODE:
      if (!CpuPreprocessor(input_msg, dstimg, stream)) {
        return false;
      }
      break;
    default:
      break;
  }

  return true;
}

/**
 * @description: Gpu preprocessor.
 */
bool PreProcessor::GpuPreprocessor(InfertMsg& input_msg,
    float* dstimg, cudaStream_t stream) {

  checkRuntime(cudaMemcpy(input_data_device_, input_msg.image.data,\
      input_msg.img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

  auto preprocess = Registry::getInstance()->getRegisterFunc<InfertMsg&, float*,
                    uint8_t*, std::shared_ptr<ParseMsgs>&>(parsemsgs_->preprocess_type_);

  preprocess(input_msg, dstimg, input_data_device_, parsemsgs_);

  return true;
}

/**
 * @description: Cpu preprocessor.
 */
bool PreProcessor::CpuPreprocessor(InfertMsg& input_msg,
    float* dstimg, cudaStream_t stream) {

  auto preprocess = Registry::getInstance()->getRegisterFunc<InfertMsg&, float*,
                    std::shared_ptr<ParseMsgs>&>(parsemsgs_->preprocess_type_);

  preprocess(input_msg, input_data_host_, parsemsgs_);

  checkRuntime(cudaMemcpy(dstimg, input_data_host_, \
      sizeof(float) * parsemsgs_->dstimg_size_, cudaMemcpyHostToDevice));

  return true;
}

/**
 * @description: AffineMatrix.
 */
void PreProcessor::CalAffineMatrix(InfertMsg& input_msg) {
  float scale_x = parsemsgs_->dst_img_w_ / static_cast<float>(input_msg.width);
  float scale_y = parsemsgs_->dst_img_h_ / static_cast<float>(input_msg.height);
  float scale = min(scale_x, scale_y);

  input_msg.affineMatrix(0, 0) = scale;
  input_msg.affineMatrix(1, 1) = scale;
  input_msg.affineMatrix(0, 2) = -scale * input_msg.width  * 0.5 + parsemsgs_->dst_img_w_ * 0.5 + scale * 0.5 - 0.5;
  input_msg.affineMatrix(1, 2) = -scale * input_msg.height * 0.5 + parsemsgs_->dst_img_h_ * 0.5 + scale * 0.5 - 0.5;

  input_msg.affineMatrix_cv = \
      (cv::Mat_<float>(2, 3) << scale, 0.0,
      -scale * input_msg.width  * 0.5 + parsemsgs_->dst_img_w_ * 0.5 + scale * 0.5 - 0.5,
                                 0.0, scale,
      -scale * input_msg.height * 0.5 + parsemsgs_->dst_img_h_ * 0.5 + scale * 0.5 - 0.5);

  // Compute inverse
  input_msg.affineMatrix_inv = input_msg.affineMatrix.inverse();
}

/**
 * @description: Memory allocator.
 */
bool PreProcessor::MemAllocator() {
  checkRuntime(cudaMalloc(&input_data_device_, parsemsgs_->srcimg_size_));
  checkRuntime(cudaMallocHost(&input_data_host_, sizeof(float) * parsemsgs_->dstimg_size_));
  return true;
}

/**
 * @description: Cpu and gpu memory free.
 */
bool PreProcessor::MemFree() {
  checkRuntime(cudaFree(input_data_device_));
  checkRuntime(cudaFreeHost(input_data_host_));
  return true;
}

/**
 * @description: Load file.
 */
std::vector<uint8_t> PreProcessor::LoadFile(const string& file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();
  vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read(reinterpret_cast<char*>(&data[0]), length);
  }
  in.close();
  return data;
}

}  // namespace appinfer
}  // namespace hpc
