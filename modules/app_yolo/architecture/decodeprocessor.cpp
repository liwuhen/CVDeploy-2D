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
  nms_plugin_ = std::make_shared<NmsPlugin>();
  nms_plugin_->SetParam(parsemsgs_);
  nms_plugin_->Init();

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
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: DecodeProcessor module set param failed ");
    return false;
  }
  imgshape_["dst"] = make_pair(parsemsgs_->dst_img_h_, parsemsgs_->dst_img_w_);

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
bool DecodeProcessor::Inference(float* predict,
    InfertMsg& infer_msg, std::shared_ptr<InferMsgQue>& bboxQueue)
{
  imgshape_["src"] = make_pair(infer_msg.height, infer_msg.width);

  vector<Box> box_result;
  CpuDecode(predict, infer_msg, box_result);

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
void DecodeProcessor::Visualization(bool real_time,
    cv::Mat& img, int64_t timestamp, vector<Box>& results)
{
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
void DecodeProcessor::ScaleBoxes(vector<Box>& box_result)
{
  float gain  = min(imgshape_["dst"].first / static_cast<float>(imgshape_["src"].first),\
                imgshape_["dst"].second / static_cast<float>(imgshape_["src"].second));
  float pad[] = {(imgshape_["dst"].second - imgshape_["src"].second * gain) * 0.5, \
                (imgshape_["dst"].first - imgshape_["src"].first * gain) * 0.5};
  for (int index = 0; index < box_result.size(); index++) {
    box_result[index].left   = clamp((box_result[index].left - pad[0]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].second));
    box_result[index].right  = clamp((box_result[index].right - pad[0]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].second));
    box_result[index].top    = clamp((box_result[index].top - pad[1]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].first));
    box_result[index].bottom = clamp((box_result[index].bottom - pad[1]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].first));
  }
}

/**
 * @description: Bounding box decoding at feature level．
 */
void DecodeProcessor::BboxDecodeFeatureLevel(float* predict,
    InfertMsg& infer_msg, vector<Box>& box_result)
{
  // for (int j = 0; j < out_node_vec[1]; j++)
  // {
  //   float* lables_node = cpu_output_buffers_[0] + j * 3;
  //   float* scores_node = cpu_output_buffers_[1] + j * 1;
  //   float* boxes_node  = cpu_output_buffers_[2] + j * 4;  // 特征图级别

  //   int label  = std::max_element(lables_node, lables_node + 3) - lables_node;
  //   float prob = lables_node[label];

  //   float objness = scores_node[0];
  //   if(objness < confidence_threshold)
  //       continue;

  //   float confidence = prob * objness;
  //   if(confidence < confidence_threshold)
  //       continue;

  //   if (j < 7680) {
  //       grid_x = anchor_points[0][j].first;
  //       grid_y = anchor_points[0][j].second;
  //       stride = 8;
  //   }
  //   else if (j >= 7680 && j < 9600) {
  //       grid_x = anchor_points[1][j-7680].first;
  //       grid_y = anchor_points[1][j-7680].second;
  //       stride = 16;
  //   }
  //   else if (j >= 9600 && j < 10080) {
  //       grid_x = anchor_points[2][j-9600].first;
  //       grid_y = anchor_points[2][j-9600].second;
  //       stride = 32;
  //   }

  //   // 特征图级别 -> 输入图像层级
  //   float cx     = (boxes_node[0] + grid_x) * stride;  // 输入图像级别
  //   float cy     = (boxes_node[1] + grid_y) * stride;
  //   float width  = exp(boxes_node[2]) * stride;
  //   float height = exp(boxes_node[3]) * stride;  // anchor free
  //   float left   = cx - width  * 0.5;  // 输入图像级别
  //   float top    = cy - height * 0.5;
  //   float right  = cx + width  * 0.5;
  //   float bottom = cy + height * 0.5;

  //   // 输入图像层级 -> 原图图像层级
  //   float image_base_left   = d2i[0] * left   + d2i[2];
  //   float image_base_right  = d2i[0] * right  + d2i[2];
  //   float image_base_top    = d2i[0] * top    + d2i[5];
  //   float image_base_bottom = d2i[0] * bottom + d2i[5];
  //   bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
  // }
}

/**
 * @description: Bounding box decoding at input level．
 */
void DecodeProcessor::BboxDecodeInputLevel(float* predict,
    InfertMsg& infer_msg, vector<Box>& box_result)
{
  vector<Box> boxes;
  int num_classes = parsemsgs_->predict_dim_[2] - 5;
  for (int i = 0; i < parsemsgs_->predict_dim_[1]; ++i)
  {
    float* pitem  = predict + i * parsemsgs_->predict_dim_[2];
    float objness = pitem[4];
    if (objness < parsemsgs_->obj_threshold_) continue;
    float* pclass = pitem + 5;

    int label  = std::max_element(pclass, pclass + num_classes) - pclass;
    float prob = pclass[label];
    float confidence = prob * objness;
    if (confidence < parsemsgs_->obj_threshold_) continue;

    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];
    float left   = cx - width  * 0.5;
    float top    = cy - height * 0.5;
    float right  = cx + width  * 0.5;
    float bottom = cy + height * 0.5;

    // 输入图像层级模型预测框 ==> 映射回原图上尺寸
    float image_left   = infer_msg.affineMatrix_inv(0, 0) * left   + infer_msg.affineMatrix_inv(0, 2);
    float image_top    = infer_msg.affineMatrix_inv(1, 1) * top    + infer_msg.affineMatrix_inv(1, 2);
    float image_right  = infer_msg.affineMatrix_inv(0, 0) * right  + infer_msg.affineMatrix_inv(0, 2);
    float image_bottom = infer_msg.affineMatrix_inv(1, 1) * bottom + infer_msg.affineMatrix_inv(1, 2);

    boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
  }
  nms_plugin_->Nms(boxes, box_result, parsemsgs_->nms_threshold_);
}

/**
 * @description: Cpu decode．
 */
void DecodeProcessor::CpuDecode(float* predict,
    InfertMsg& infer_msg, vector<Box>& box_result)
{
  if((DecodeType)parsemsgs_->decode_type_ == DecodeType::FEATURE_LEVEL) {
    BboxDecodeFeatureLevel(predict, infer_msg, box_result);
  } else if ((DecodeType)parsemsgs_->decode_type_ == DecodeType::INPUT_LEVEL) {
    BboxDecodeInputLevel(predict, infer_msg, box_result);
  } else {
    GLOG_ERROR("[CpuDecode]: Decoding method error. ");
  }
}

}  // namespace appinfer
}  // namespace hpc
