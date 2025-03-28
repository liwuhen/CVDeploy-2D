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
#ifndef APP_YOLO_NMS_REGISTRY_H__
#define APP_YOLO_NMS_REGISTRY_H__

#include <opencv2/opencv.hpp>
#include "logger.h"
#include "parseconfig.h"
#include "task_struct.hpp"
#include "function_registry.hpp"

namespace hpc {

namespace appinfer {


inline float calIou(Box& box_a, Box& box_b) {
  // iou 计算
  double eps = 1e-7;
  float iou_result   = 0.0f;
  float cross_left   = std::max(box_a.left, box_b.left);
  float cross_top    = std::max(box_a.top, box_b.top);
  float cross_right  = std::min(box_a.right, box_b.right);
  float cross_bottom = std::min(box_a.bottom, box_b.bottom);

  float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
  float union_area =
      std::max(0.0f, box_a.right - box_a.left) * std::max(0.0f, box_a.bottom - box_a.top) +\
      std::max(0.0f, box_b.right - box_b.left) * std::max(0.0f, box_b.bottom - box_b.top) - cross_area + eps;
  iou_result = cross_area / union_area;

  return iou_result;
}

/**
 * @description: Nms.
 */
inline void CalNms(
    float nms_threshold,
    vector<Box>& boxes,
    vector<Box>& box_result) {

    // nms 过程：
    // 1、按照类别概率排序，2、从最大概率矩形框开始，与其他框判断 iou
    // 是否超过阈值，3、标记重叠度超过阈值的框，丢掉，4、从剩余框中选择概率最大框并于剩余框判断重叠度是否超过阈值，重复步骤
    // 3 过程
    std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b) { return a.confidence > b.confidence; });  // 这里给的引用，可以避免拷贝。从大到小排序
    std::vector<bool> remove_flags(boxes.size());                                                       // 初始化容器容量，并预先分配内存。
    box_result.reserve(boxes.size());                                                                   // reserve 主动分配内存可以提升程序执行效率

    for (int i = 0; i < boxes.size(); ++i) {
        if (remove_flags[i]) continue;

        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) continue;

            auto& jbox = boxes[j];
            if (ibox.label == jbox.label) {
                // class matched
                if (calIou(ibox, jbox) >= nms_threshold) remove_flags[j] = true;
            }
        }
    }

}

// 全局自动注册
REGISTER_CALIBRATOR_FUNC("nms", CalNms);

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_NMS_REGISTRY_H__
