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
#ifndef APP_YOLO_POSTPROCESS_REGISTRY_H__
#define APP_YOLO_POSTPROCESS_REGISTRY_H__

#include <opencv2/opencv.hpp>
#include "logger.h"
#include "parseconfig.h"
#include "nms_registry.hpp"

namespace hpc {

namespace appinfer {

/**
 * @description: YOLOV5 cpu postprocess anchor base.
 */
inline void PostprocessV5CpuAchorBase(
    InfertMsg& infer_msg,
    std::vector<Box>& box_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    vector<Box> boxes;
    int num_classes = parsemsgs->predict_dim_[0][2] - 5;
    for (int i = 0; i < parsemsgs->predict_dim_[0][1]; ++i)
    {
        float* pitem  = predict[0] + i * parsemsgs->predict_dim_[0][2];
        float* pclass = pitem + 5;

        float objitem = pitem[4];
        if ( objitem < parsemsgs->obj_threshold_ ) continue;

        int label  = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob * objitem;    // anchor free
        if (confidence < parsemsgs->obj_threshold_) continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width  * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width  * 0.5;
        float bottom = cy + height * 0.5;

        // 输入图像层级模型预测框 ==> 映射回原图上尺寸
        float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);
        float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);

        if ( image_left < 0 || image_top< 0 ) {
            continue;
        }

        boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
    }

    auto nms = Registry::getInstance()->getRegisterFunc<float,
                std::vector<Box>&, std::vector<Box>&>(parsemsgs->nms_type_);

    nms(parsemsgs->nms_threshold_, boxes, box_result);

}

/**
 * @description: YOLOV5 cpu postprocess anchor free.
 */
inline void PostprocessV5CpuAchorFree(
    InfertMsg& infer_msg,
    std::vector<Box>& box_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    vector<Box> boxes;
    int num_classes = parsemsgs->predict_dim_[0][2] - 4;
    for (int i = 0; i < parsemsgs->predict_dim_[0][1]; ++i)
    {
        float* pitem  = predict[0] + i * parsemsgs->predict_dim_[0][2];
        float* pclass = pitem + 4;

        int label  = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob;    // anchor free
        if (confidence < parsemsgs->obj_threshold_) continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width  * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width  * 0.5;
        float bottom = cy + height * 0.5;

        // 输入图像层级模型预测框 ==> 映射回原图上尺寸
        float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);
        float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);

        if ( image_left < 0 || image_top< 0 ) {
            continue;
        }

        boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
    }

    auto nms = Registry::getInstance()->getRegisterFunc<float,
                std::vector<Box>&, std::vector<Box>&>(parsemsgs->nms_type_);

    nms(parsemsgs->nms_threshold_, boxes, box_result);

}

/**
 * @description: YOLOV8 cpu postprocess anchor free.
 */
inline void PostprocessV8CpuAchorFree(
    InfertMsg& infer_msg,
    std::vector<Box>& box_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    vector<Box> boxes;
    int num_classes = parsemsgs->predict_dim_[0][2] - 4;
    for (int i = 0; i < parsemsgs->predict_dim_[0][1]; ++i)
    {
        float* pitem  = predict[0] + i * parsemsgs->predict_dim_[0][2];
        float* pclass = pitem + 4;

        int label  = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob;    // anchor free
        if (confidence < parsemsgs->obj_threshold_) continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width  * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width  * 0.5;
        float bottom = cy + height * 0.5;

        // 输入图像层级模型预测框 ==> 映射回原图上尺寸
        float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);
        float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);

        if ( image_left < 0 || image_top< 0 ) {
            continue;
        }

        boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
    }

    auto nms = Registry::getInstance()->getRegisterFunc<float,
                std::vector<Box>&, std::vector<Box>&>(parsemsgs->nms_type_);

    nms(parsemsgs->nms_threshold_, boxes, box_result);

}

/**
 * @description: YOLOV11 cpu postprocess anchor free.
 */
inline void PostprocessV11CpuAchorFree(
    InfertMsg& infer_msg,
    std::vector<Box>& box_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    vector<Box> boxes;
    int num_classes = parsemsgs->predict_dim_[0][2] - 4;
    for (int i = 0; i < parsemsgs->predict_dim_[0][1]; ++i)
    {
        float* pitem  = predict[0] + i * parsemsgs->predict_dim_[0][2];
        float* pclass = pitem + 4;

        int label  = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob;    // anchor free
        if (confidence < parsemsgs->obj_threshold_) continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width  * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width  * 0.5;
        float bottom = cy + height * 0.5;

        // 输入图像层级模型预测框 ==> 映射回原图上尺寸
        float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);
        float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);

        if ( image_left < 0 || image_top< 0 ) {
            continue;
        }

        boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
    }

    auto nms = Registry::getInstance()->getRegisterFunc<float,
                std::vector<Box>&, std::vector<Box>&>(parsemsgs->nms_type_);

    nms(parsemsgs->nms_threshold_, boxes, box_result);

}

// 全局自动注册
REGISTER_CALIBRATOR_FUNC("postv5_cpu_anchorbase", PostprocessV5CpuAchorBase);
REGISTER_CALIBRATOR_FUNC("postv5_cpu_anchorfree", PostprocessV5CpuAchorFree);
REGISTER_CALIBRATOR_FUNC("postv8_cpu_anchorfree", PostprocessV8CpuAchorFree);
REGISTER_CALIBRATOR_FUNC("postv11_cpu_anchorfree", PostprocessV11CpuAchorFree);

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_POSTPROCESS_REGISTRY_H__
