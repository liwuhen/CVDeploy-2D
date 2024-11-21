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

#include "decodeprocessor.h"

namespace hpc {

namespace appinfer {

DecodeProcessor::DecodeProcessor() {}

DecodeProcessor::~DecodeProcessor() {}

/**
 * @description: init．
 */
bool DecodeProcessor::Init() {
  GLOG_INFO("[Init]: DecodeProcessor module init ");
  return true;
}

/**
 * @brief The inference algorithm handles threads．
 */
bool DecodeProcessor::RunStart() {
  GLOG_INFO("[RunStart]: DecodeProcessor module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool DecodeProcessor::RunStop() {
  GLOG_INFO("[RunStop]: DecodeProcessor module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool DecodeProcessor::RunRelease() {
  GLOG_INFO("[RunRelease]: DecodeProcessor module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool DecodeProcessor::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  }
  imgshape_["dst"] = make_pair<int&, int&>(parsemsgs_->dst_img_h_, parsemsgs_->dst_img_w_);

  GLOG_INFO("[SetParam]: DecodeProcessor module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool DecodeProcessor::DataResourceRelease() {}

/**
 * @description: Inference
 */
bool DecodeProcessor::Inference(float* predict, InfertMsg& infer_msg,
                                std::shared_ptr<InferMsgQue>& bboxQueue) {
  imgshape_["src"] = make_pair<int, int>(infer_msg.height, infer_msg.width);

  vector<Box> box_result;
  CpuDecode(predict, box_result);

  InfertMsg msg;
  msg = infer_msg;
  for (auto& box : box_result) {
    msg.bboxes.emplace_back(box);
  }
  bboxQueue->Push(msg);

  Visualization(false, infer_msg.image, infer_msg.frame_id, box_result);

  return true;
}

/**
 * @description: Visualization
 */
void DecodeProcessor::Visualization(bool real_time, cv::Mat& img, int64_t timestamp,
                                    vector<Box>& results) {
  for (auto& box : results) {
    cv::Scalar color;
    tie(color[0], color[1], color[2]) = random_color(box.label);
    auto name = cocolabels[box.label];
    auto caption = cv::format("%s %.2f", name, box.confidence);
    cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);
    cv::putText(img, caption, cv::Point(box.left, box.top - 7), 0, 0.8, color, 2, 16);
  }

  if (real_time) {
    cv::imshow("Live Video", img);
    // 按 'q' 键退出
    if (cv::waitKey(30) >= 0) {
      return;
    }
  } else {
    std::string path = parsemsgs_->save_img_ + "/img_" + std::to_string(timestamp) + ".jpg";
    cv::imwrite(path, img);
  }
}

/**
 * @description: Bbox mapping to original map scale.
 */
void DecodeProcessor::ScaleBoxes(vector<Box>& box_result) {
  float gain = min(imgshape_["dst"].first / static_cast<float>(imgshape_["src"].first),
                   imgshape_["dst"].second / static_cast<float>(imgshape_["src"].second));
  float pad[] = {(imgshape_["dst"].second - imgshape_["src"].second * gain) * 0.5,
                 (imgshape_["dst"].first - imgshape_["src"].first * gain) * 0.5};
  for (int index = 0; index < box_result.size(); index++) {
    box_result[index].left = clamp((box_result[index].left - pad[0]) / gain, 0.0f,
                                   static_cast<float>(imgshape_["src"].second));
    box_result[index].right = clamp((box_result[index].right - pad[0]) / gain, 0.0f,
                                    static_cast<float>(imgshape_["src"].second));
    box_result[index].top = clamp((box_result[index].top - pad[1]) / gain, 0.0f,
                                  static_cast<float>(imgshape_["src"].first));
    box_result[index].bottom = clamp((box_result[index].bottom - pad[1]) / gain, 0.0f,
                                     static_cast<float>(imgshape_["src"].first));
  }
}

/**
 * @description: Nms.
 */
void DecodeProcessor::Nms(vector<Box>& boxes, vector<Box>& box_result, float nms_threshold) {
  // nms 过程：
  // 1、按照类别概率排序，2、从最大概率矩形框开始，与其他框判断 iou
  // 是否超过阈值，3、标记重叠度超过阈值的框，丢掉，4、从剩余框中选择概率最大框并于剩余框判断重叠度是否超过阈值，重复步骤
  // 3 过程
  std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b) {
    return a.confidence > b.confidence;
  });  // 这里给的引用，可以避免拷贝。从大到小排序
  std::vector<bool> remove_flags(boxes.size());  // 初始化容器容量，并预先分配内存。
  box_result.reserve(boxes.size());  // reserve 主动分配内存可以提升程序执行效率

  // iou 计算
  auto iou = [](const Box& a, const Box& b) {
    float cross_left = std::max(a.left, b.left);
    float cross_top = std::max(a.top, b.top);
    float cross_right = std::min(a.right, b.right);
    float cross_bottom = std::min(a.bottom, b.bottom);

    float cross_area =
        std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
    float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) +
                       std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) -
                       cross_area;
    if (cross_area == 0 || union_area == 0) return 0.0f;
    return cross_area / union_area;
  };

  for (int i = 0; i < boxes.size(); ++i) {
    if (remove_flags[i]) continue;

    auto& ibox = boxes[i];
    box_result.emplace_back(ibox);
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (remove_flags[j]) continue;

      auto& jbox = boxes[j];
      if (ibox.label == jbox.label) {
        // class matched
        if (iou(ibox, jbox) >= nms_threshold) remove_flags[j] = true;
      }
    }
  }
}

/**
 * @description: Cpu decode．
 */
void DecodeProcessor::CpuDecode(float* predict, vector<Box>& box_result) {
  vector<Box> boxes;
  int num_classes = parsemsgs_->predict_dim_[2] - 5;
  for (int i = 0; i < parsemsgs_->predict_dim_[1]; ++i) {
    // pitem 每次偏移 85 个元素，154 字节（容器中元素为 float 类型 4 个字节），pitem 指向 tensor
    // 中每一行的首地址。
    float* pitem =
        predict + i * parsemsgs_->predict_dim_[2];  // predict 是 yolov5 输出 tensor 的首地址，后面
                                                    // i*cols 为内存地址的偏移量
    float objness = pitem
        [4];  // 这里的 pitem
              // 是一段连续地址的首地址，通过索引的方式（索引号表示一个元素<==>内存地址偏移一个元素的类型字节数）<==>内存地址的偏移（偏移量为一个元素的类型字节数）
    if (objness < parsemsgs_->obj_threshold_) continue;

    float* pclass = pitem + 5;
    // max_element：获得最大值的地址。
    int label =
        std::max_element(pclass, pclass + num_classes) -
        pclass;  // pclass + num_classes 表示偏移 80 个内存位置（80 个 float 类型的字节大小）。
    float prob = pclass[label];  // 等价于内存的偏移，并索取对应内存地址的元素值。
    float confidence = prob * objness;
    if (confidence < parsemsgs_->obj_threshold_) continue;

    float cx = pitem[0];
    float cy = pitem[1];
    float width = pitem[2];
    float height = pitem[3];
    float left = cx - width * 0.5;
    float top = cy - height * 0.5;
    float right = cx + width * 0.5;
    float bottom = cy + height * 0.5;
    boxes.emplace_back(left, top, right, bottom, confidence,
                       label);  // 减少构造函数的调用和内存的拷贝
  }

  Nms(boxes, box_result, parsemsgs_->nms_threshold_);

  // 模型输入图像上的模型预测框 ==> 映射回原图上尺寸
  ScaleBoxes(box_result);
}

}  // namespace appinfer
}  // namespace hpc
