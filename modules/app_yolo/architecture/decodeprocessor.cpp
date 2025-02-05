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
  model_decode_ = std::make_shared<ModelDecode>();
  model_decode_->SetParam(parsemsgs_);
  model_decode_->Init();

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
bool DecodeProcessor::Inference(std::vector<float*>& predict,
    InfertMsg& infer_msg, std::shared_ptr<InferMsgQue>& bboxQueue) {
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
    cv::Mat& img, int64_t timestamp, vector<Box>& results) {
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
 * @description: Cpu decode．
 */
void DecodeProcessor::CpuDecode(std::vector<float*>& predict,
    InfertMsg& infer_msg, vector<Box>& box_result) {

  if((DecodeType)parsemsgs_->decode_type_ == DecodeType::FEATURE_LEVEL) {
    model_decode_->BboxDecodeFeatureLevel(predict, infer_msg, box_result);
  } else if ((DecodeType)parsemsgs_->decode_type_ == DecodeType::INPUT_LEVEL) {
    model_decode_->BboxDecodeInputLevel(predict, infer_msg, box_result);
  } else {
    GLOG_ERROR("[CpuDecode]: Decoding method error. ");
  }
}

}  // namespace appinfer
}  // namespace hpc
