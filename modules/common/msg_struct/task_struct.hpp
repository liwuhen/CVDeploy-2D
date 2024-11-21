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

#ifndef APP_COMMON_TASK_STRUCT_H__
#define APP_COMMON_TASK_STRUCT_H__

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Eigen/Dense"
#include "common.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "std_buffer.h"

namespace hpc {

namespace common {

using namespace std;
using namespace Eigen;

struct Box {
  float left, top, right, bottom, confidence;
  int label;

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int label)
      : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}

  Box operator=(const Box& obj) {
    this->left = obj.left;
    this->top = obj.top;
    this->right = obj.right;
    this->bottom = obj.bottom;
    this->confidence = obj.confidence;
    this->label = obj.label;
    return *this;
  }
};

struct InfertMsg {
  InfertMsg() : width(0), height(0), frame_id(0), timestamp(0), img_size(0) {
    this->image = cv::Mat();
  }

  InfertMsg operator=(const InfertMsg& obj) {
    this->width = obj.width;
    this->height = obj.height;
    this->img_size = obj.img_size;
    this->frame_id = obj.frame_id;
    this->timestamp = obj.timestamp;
    this->image = obj.image.clone();
    this->affineMatrix = obj.affineMatrix;
    this->affineMatrix_inv = obj.affineMatrix_inv;

    for (auto& box : obj.bboxes) {
      this->bboxes.emplace_back(box);
    }
    return *this;
  }

 public:
  uint32_t width;
  uint32_t height;
  uint32_t img_size;
  uint32_t frame_id;
  int64_t timestamp;
  cv::Mat image;
  std::vector<Box> bboxes;
  Eigen::Matrix3f affineMatrix = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f affineMatrix_inv = Eigen::Matrix3f::Identity();
};

// 自定义图像，去除opencv
struct CVImage {
  int img_width;
  int img_height;
  int img_channel;
  int img_size;
  std::vector<uint8_t> data;  // 存储连续的像素值

  CVImage();
  CVImage(int width, int height, int channel) : img_width(width), \
    img_height(height), img_channel(channel), img_size(0) {}

  ~CVImage() {}

  CVImage operator=(const CVImage& obj) {
    this->img_width = obj.img_width;
    this->img_height = obj.img_height;
    this->img_channel = obj.img_channel;
    if (!data.empty()) {
      auto length = obj.img_width * obj.img_height * obj.img_channel;
      std::memcpy(data.data(), obj.data.data(), length);  // 拷贝数据
    }
    return *this;
  }

  // 指针类型
  void setDataFromArray(const uint8_t* obj, int size) {
    img_size = img_width * img_height * img_channel;
    if (obj != nullptr && size <= img_size) {
      std::memcpy(data.data(), obj, size);  // 拷贝数据
    }
  }
};

using InputMsgQue = MsgQueue<InfertMsg>;

using InferMsgQue = MsgQueue<InfertMsg>;

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_TASK_STRUCT_H__
