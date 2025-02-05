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

#include "postprocessor.h"

namespace hpc {

namespace appinfer {

PostProcessor::PostProcessor() {}

PostProcessor::~PostProcessor() {}

/**
 * @description: init．
 */
bool PostProcessor::Init() {
  GLOG_INFO("[Init]: PostProcessor module init ");
  return true;
}

/**
 * @brief The inference algorithm handles threads．
 */
bool PostProcessor::RunStart() {
  running_.store(true);

  inference_ =
      std::dynamic_pointer_cast<InferenceEngine>(GetModulePipeline(ModuleType::MODULE_INFERENCE));

  worker_ptr_ = std::make_shared<std::thread>([this]() { Inference(); });

  GLOG_INFO("[RunStart]: PostProcessor module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool PostProcessor::RunStop() {
  while (!inference_->bboxQueue_->Empty()) {
  }

  running_.store(false);
  if (worker_ptr_->joinable()) {
    worker_ptr_->join();
  }

  DataResourceRelease();

  GLOG_INFO("[RunStop]: PostProcessor module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool PostProcessor::RunRelease() {
  running_.store(false);
  if (worker_ptr_->joinable()) {
    worker_ptr_->join();
  }

  DataResourceRelease();

  GLOG_INFO("[RunRelease]: PostProcessor module release");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool PostProcessor::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  }

  GLOG_INFO("[SetParam]: PostProcessor module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool PostProcessor::DataResourceRelease() {
  inference_->bboxQueue_->Clear();
  return true;
}

/**
 * @description: Inference.
 */
void PostProcessor::Inference() {
  while (running_.load()) {
    if (!inference_->bboxQueue_->Pop(output_msg_)) {
      continue;
    }
  }

  running_.store(false);
}

/**
 * @description: Visualization.
 */
void PostProcessor::Visualization(bool real_time, cv::Mat& img, int64_t timestamp,
                                  vector<Box>& results) {
  for (auto& box : results) {
    cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom),
                  cv::Scalar(0, 255, 0), 2);
    cv::putText(img, cv::format("%.2f", box.confidence), cv::Point(box.left, box.top - 7), 0, 0.8,
                cv::Scalar(0, 0, 255), 2, 16);
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

}  // namespace appinfer
}  // namespace hpc
