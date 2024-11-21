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

#ifndef APP_COMMON_GLOG_H__
#define APP_COMMON_GLOG_H__

#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <mutex>

#include "parseconfig.h"

namespace hpc {
namespace common {

class GlogMsg {
 public:
  static GlogMsg& GetInstance() {
    static GlogMsg instance;
    return instance;
  }

  void Init(int level) {
    ParseMsgs_ = std::make_shared<ParseMsgs>();

    std::lock_guard<std::mutex> lock(glog_mutex_);
    if (init_) return;
    FLAGS_logtostderr = false;
    FLAGS_log_prefix = true;
    FLAGS_logbufsecs = 0;
    FLAGS_max_log_size = 15;
    FLAGS_stop_logging_if_full_disk = true;
    FLAGS_stderrthreshold = level;
    FLAGS_minloglevel = level;

    google::SetLogDestination(google::GLOG_INFO, ParseMsgs_->log_path_.c_str());
    google::SetLogDestination(google::GLOG_WARNING, "");
    google::SetLogDestination(google::GLOG_ERROR, "");
    google::SetLogDestination(google::GLOG_FATAL, "");
    google::SetLogSymlink(google::GLOG_INFO, ParseMsgs_->log_path_.c_str());

    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_dir = ParseMsgs_->log_path_.c_str();
    google::InitGoogleLogging("APP_AUTOPILOT");
    init_ = true;
  }

 private:
  GlogMsg() = default;
  ~GlogMsg() {
    google::FlushLogFiles(google::GLOG_INFO);
    google::ShutdownGoogleLogging();
  };
  bool init_ = false;
  std::mutex glog_mutex_;
  std::shared_ptr<ParseMsgs> ParseMsgs_;
};

inline void InitLogging(int level = 0) { GlogMsg::GetInstance().Init(level); }

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_GLOG_H__
