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

#include "inference.h"

namespace hpc {

namespace appinfer {

InferenceEngine::InferenceEngine() {}

InferenceEngine::~InferenceEngine() {}

/**
 * @description: init．
 */
bool InferenceEngine::Init() {
  this->trtInfer_  = std::make_shared<TrtInfer>();
  preProcessor_    = std::make_shared<PreProcessor>();
  decodeProcessor_ = std::make_shared<DecodeProcessor>();
  trtInfer_->SetParam(parsemsgs_);
  preProcessor_->SetParam(parsemsgs_);
  decodeProcessor_->SetParam(parsemsgs_);
  trtInfer_->Init();
  preProcessor_->Init();
  decodeProcessor_->Init();

  {
    this->output_img_device_ = nullptr;
    inputMsgQue_ = std::make_shared<InputMsgQue>(parsemsgs_->input_msgdepth_);
    bboxQueue_   = std::make_shared<InferMsgQue>(parsemsgs_->decode_msgdepth_);
    checkRuntime(cudaMalloc(&output_img_device_, sizeof(float) * parsemsgs_->dstimg_size_));
  }

  GLOG_INFO("[Init]: Inference module init");
  return true;
}

/**
 * @description: The inference algorithm handles threads．
 */
bool InferenceEngine::RunStart() {
  running_.store(true);

  callbackMsg.clear();
  bboxQueue_->Clear();
  inputMsgQue_->Clear();

  worker_ptr_ = std::make_shared<std::thread>([this]() { Inference(); });

  GLOG_INFO("[RunStart]: Inference module start");
  return true;
}

/**
 * @description: Thread stop．
 */
bool InferenceEngine::RunStop() {
  while (!inputMsgQue_->Empty()) {
  }

  running_.store(false);
  if (worker_ptr_->joinable()) {
    worker_ptr_->join();
  }

  GLOG_INFO("[RunStop]: Inference module stop");
  return true;
}

/**
 * @description: Software function stops．
 */
bool InferenceEngine::RunRelease() {
  running_.store(false);
  if (worker_ptr_->joinable()) {
    worker_ptr_->join();
  }

  DataResourceRelease();

  GLOG_INFO("[RunRelease]: Inference module release");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool InferenceEngine::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  }

  GLOG_INFO("[SetParam]: Inference module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool InferenceEngine::DataResourceRelease() {
  callbackMsg.clear();
  inputMsgQue_->Clear();

  if (!preProcessor_->MemFree()) return false;
  if (!trtInfer_->MemFree()) return false;

  if (!checkRuntime(cudaFree(output_img_device_))) return false;
  return true;
}

/**
 * @description: Load data.
 */
bool InferenceEngine::LoadData(InfertMsg& infer_msg) {
  inputMsgQue_->Push(infer_msg);
  return true;
}

/**
 * @description: Timer.
 */
void InferenceEngine::TimeManager(int interval, std::function<void()> task) {}

/**
 * @description: Inference.
 */
bool InferenceEngine::Inference() {
  while (running_.load()) {
    if (!inputMsgQue_->Pop(input_msg_)) {
      continue;
    }

    GLOG_INFO("[CV-Deploy]: Process frame: " + std::to_string(input_msg_.index));

    // preprocessor
    auto start_timer = timestamp_now_float();
    auto begin_timer = timestamp_now_float();
    if (!preProcessor_->Inference(input_msg_, output_img_device_, \
        (DeviceMode)parsemsgs_->infer_mode_, nullptr)) {
      GLOG_ERROR("PreProcessor module error. ");
      return false;
    }
    preproccess_time_.push_back(timestamp_now_float() - begin_timer);

    // inference
    begin_timer = timestamp_now_float();
    if (!trtInfer_->Inference(output_img_device_)) {
      GLOG_ERROR("TRT inference error. ");
      return false;
    }
    infer_time_.push_back(timestamp_now_float() - begin_timer);

    // decode bbox
    begin_timer = timestamp_now_float();
    if (!decodeProcessor_->Inference(trtInfer_->cpu_buffers_, \
        input_msg_, callbackMsg, bboxQueue_)) {
      GLOG_ERROR("Decode module error. ");
      return false;
    }
    decode_time_.push_back(timestamp_now_float() - begin_timer);
    endtoend_time_.push_back(timestamp_now_float() - start_timer);
  }

  // time info
  {
    // FPS && Times
    auto averageTimeVec = GetAverageTime(preproccess_time_, infer_time_, decode_time_, endtoend_time_);
    auto average_fps    = input_msg_.index / (averageTimeVec[4] / 1000.0);
    GLOG_INFO("Average preProcessor ms: "<< averageTimeVec[0] << " inference ms: "<< averageTimeVec[1] \
      << "\n decode ms: "<<averageTimeVec[2]<<" end to end ms: "<<averageTimeVec[3]<< " fps: "<<average_fps);
  }
  running_.store(false);

  return true;
}

/**
 * @description: Python model output binary for post-processing inference.
 */
bool InferenceEngine::InferenceV1() {
  // 数据处理
  LOG(INFO) << " Begin inferenceV1 ... ";
  auto image = cv::imread(parsemsgs_->img_path_);
  auto data = trtInfer_->LoadFile(parsemsgs_->predict_path_);
  float* ptr = reinterpret_cast<float*>(data.data());
  int nelem = data.size() / sizeof(float);
  int nrows = nelem / 6.0;  // 行数，预测的 object 的数量

  // InfertMsg msg;
  // msg.width     = image.cols;
  // msg.height    = image.rows;
  // msg.frame_id  = 0;
  // msg.timestamp = GetSystmeTime();
  // msg.image     = image.clone();

  // bboxQueue_->Push(msg);

  return true;
}

void InferenceEngine::TriggerCallback(vector<InfertMsg>& infer_msg_vec) {
  if ( !running_ ) {
    infer_msg_vec.insert(infer_msg_vec.end(), callbackMsg.begin(), callbackMsg.end());
  }
}

}  // namespace appinfer
}  // namespace hpc
