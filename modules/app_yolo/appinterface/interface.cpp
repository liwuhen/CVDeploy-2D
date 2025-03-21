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

#include "interface.h"

#include <iostream>

#include "appconfig.h"
#include "inferpipeline.h"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;

class InterfaceYolo::InferInterFace {
 public:
  InferInterFace() { inferpipeline_ = std::make_shared<InferPipeline>(); }
  ~InferInterFace() {}

 public:
  /**
   * @brief     init．
   * @param[in] void．
   * @return    bool.
   */
  bool Init() { return inferpipeline_->Init(); }

  /**
   * @brief     The inference algorithm handles threads．
   * @param[in] void．
   * @return    bool.
   */
  bool Start() { return inferpipeline_->Start(); }

  /**
   * @brief     Thread stop．
   * @param[in] void．
   * @return    bool.
   */
  bool Stop() { return inferpipeline_->Stop(); }

  /**
   * @brief     Software function stops．
   * @param[in] void．
   * @return    bool.
   */
  bool Release() { return inferpipeline_->Release(); }

  /**
   * @brief     Load data.
   * @param[in] template <typename T>&.
   * @return    bool.
   */
  template <typename T>
  bool LoadData(T& class_t) {
    return inferpipeline_->LoadData(class_t);
  }

  /**
   * @brief     Load data.
   * @param[in] template <typename T>&.
   * @return    bool.
   */
  template <typename T>
  bool Callback(T& data) {
    return inferpipeline_->TriggerCallback(data);
  }

 private:
  std::shared_ptr<InferPipeline> inferpipeline_;
};

/**
 * @brief init．
 */
bool InterfaceYolo::Init() {
  infer_ = std::make_shared<InferInterFace>();
  return infer_->Init();
}

/**
 * @brief The inference algorithm handles threads．
 */
bool InterfaceYolo::Start() { return infer_->Start(); }

/**
 * @brief Thread stop．
 */
bool InterfaceYolo::Stop() { return infer_->Stop(); }

/**
 * @brief Software function stops,Release Resources．
 */
bool InterfaceYolo::Release() { return infer_->Release(); }

/**
 * @brief Load Data．
 */
template <typename T>
bool InterfaceYolo::LoadData(T& class_t) {
  return infer_->LoadData(class_t);
}

template bool InterfaceYolo::LoadData(InfertMsg& infer_msg);

/**
 * @brief Callback infer result．
 */
template <typename T>
bool InterfaceYolo::Callback(T& data) {
  return infer_->Callback(data);
}

template bool InterfaceYolo::Callback(vector<InfertMsg>& infer_msg_vec);

/**
 * @brief Init config．
 */
template <typename... Args>
bool InterfaceYolo::InitConfig(Args&&... args) {
  return InitAppConfig(std::forward<Args>(args)...);
}

template bool InterfaceYolo::InitConfig<std::string&, std::string&>(std::string&, std::string&);

}  // namespace appinfer
}  // namespace hpc
