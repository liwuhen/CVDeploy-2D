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
bool PreProcessor::Inference(InfertMsg& input_msg, float* dstimg, DeviceMode inferMode,
      cudaStream_t stream) {
  CalAffineMatrix(input_msg);

  switch (inferMode) {
    case DeviceMode::GPU_MODE:
      if (!GpuPreprocessor(input_msg, dstimg, stream)) {
        return false;
      }
      break;
    case DeviceMode::CPU_MODE:
      if (!CpuPreprocessor(input_msg.image, input_msg.timestamp, dstimg, stream))
      { return false; }
      break;
    default:
      break;
  }

  return true;
}

/**
 * @description: Gpu preprocessor.
 */
bool PreProcessor::GpuPreprocessor(InfertMsg& input_msg, float* dstimg, cudaStream_t stream) {
  checkRuntime(cudaMemcpy(input_data_device_, input_msg.image.data,
                          input_msg.img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

  if (std::string(MODEL_FLAG) == "yolov5") {
    warp_affine_bilinear(input_data_device_, parsemsgs_->batchsizes_, input_msg, dstimg,
                         parsemsgs_->dst_img_w_, parsemsgs_->dst_img_h_, 114, nullptr,
                         AppYolo::YOLOV5_MODE);
  } else if (std::string(MODEL_FLAG) == "yolox") {
    warp_affine_bilinear(input_data_device_, parsemsgs_->batchsizes_, input_msg, dstimg,
                         parsemsgs_->dst_img_w_, parsemsgs_->dst_img_h_, 114, nullptr,
                         AppYolo::YOLOX_MODE);
  } else {
  }

  return true;
}

/**
 * @description: Cpu preprocessor.
 */
bool PreProcessor::CpuPreprocessor(
    cv::Mat& srcimg,
    uint64_t timestamp,
    float* input_device_gpu,
    cudaStream_t stream) {

  checkRuntime(cudaMallocHost(&input_data_host_, sizeof(float) * parsemsgs_->dstimg_size_));

  float scale_x = parsemsgs_->dst_img_w_ / static_cast<float>(parsemsgs_->src_img_w_);
  float scale_y = parsemsgs_->dst_img_h_ / static_cast<float>(parsemsgs_->src_img_h_);
  float scale = std::min(scale_x, scale_y);
  float i2d[6], d2i[6];
  // resize 图像，源图像和目标图像几何中心的对齐
  i2d[0] = scale;
  i2d[1] = 0;
  i2d[2] = (-scale * parsemsgs_->src_img_w_ + parsemsgs_->dst_img_w_ + scale - 1) * 0.5;
  i2d[3] = 0;
  i2d[4] = scale;
  i2d[5] = (-scale * parsemsgs_->src_img_h_ + parsemsgs_->dst_img_h_ + scale - 1) * 0.5;

  cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);            // image to dst(network), 2x3 matrix
  cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);            // dst to image, 2x3 matrix
  cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // 计算一个反仿射变换

  cv::Mat input_image(parsemsgs_->dst_img_h_, parsemsgs_->dst_img_w_, CV_8UC3);
  // 对图像做平移缩放旋转变换，可逆
  cv::warpAffine(srcimg, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar::all(114));
  std::string path = parsemsgs_->save_img_ + "/img_cpu_test_" + std::to_string(timestamp) + ".jpg";
  cv::imwrite(path, input_image);

  int image_area = input_image.cols * input_image.rows;
  unsigned char* pimage = input_image.data;
  float* phost_b = input_data_host_ + image_area * 0;
  float* phost_g = input_data_host_ + image_area * 1;
  float* phost_r = input_data_host_ + image_area * 2;
  for (int i = 0; i < image_area; ++i, pimage += 3) {
    // 注意这里的顺序 rgb 调换了
    *phost_r++ = pimage[0] / 255.0f;
    *phost_g++ = pimage[1] / 255.0f;
    *phost_b++ = pimage[2] / 255.0f;
  }

  checkRuntime(cudaMemcpyAsync(input_device_gpu, input_data_host_,
                               sizeof(float) * parsemsgs_->dstimg_size_, cudaMemcpyHostToDevice,
                               stream));

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
  input_msg.affineMatrix(0, 2) =
      (-scale * input_msg.width + parsemsgs_->dst_img_w_ + scale - 1) * 0.5;
  input_msg.affineMatrix(1, 2) =
      (-scale * input_msg.height + parsemsgs_->dst_img_h_ + scale - 1) * 0.5;

  // Compute inverse
  input_msg.affineMatrix_inv = input_msg.affineMatrix.inverse();
}

/**
 * @description: Memory allocator.
 */
bool PreProcessor::MemAllocator() {
  checkRuntime(cudaMalloc(&input_data_device_, parsemsgs_->srcimg_size_));

  return true;
}

/**
 * @description: Cpu and gpu memory free.
 */
bool PreProcessor::MemFree() {
  checkRuntime(cudaFree(input_data_device_));

  return true;
}

}  // namespace appinfer
}  // namespace hpc
