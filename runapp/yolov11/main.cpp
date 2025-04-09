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

#include <iostream>
#include <opencv2/opencv.hpp>

#include "common/appconfig.h"
#include "interface.h"
#include "std_time.h"
#include "cocotojson.hpp"
#include "task_struct.hpp"

using namespace hpc::appinfer;
using namespace hpc::common;

void GetFileName(std::string& path, std::vector<std::string>& filenames) {
  DIR* pDir;
  struct dirent* ptr;
  if (!(pDir = opendir(path.c_str()))) {
    std::cout << "Folder doesn't exist!";
    return;
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      filenames.push_back(path + "/" + ptr->d_name);
    }
  }
  closedir(pDir);
}

void TestDemo(InterfaceYolo* inference, std::string& path) {
  std::vector<std::string> imgfilevec;
  GetFileName(path, imgfilevec);
  sort(imgfilevec.begin(), imgfilevec.end());
  for (int index = 0; index < imgfilevec.size(); index++) {
    auto img = cv::imread(imgfilevec[index]);
    auto file_name = FileSystem::file_name(imgfilevec[index], false);
    uint32_t img_id= atoi(file_name.c_str());

    InfertMsg msg;
    msg.width     = img.cols;
    msg.height    = img.rows;
    msg.img_size  = img.cols * img.rows * 3;
    msg.frame_id  = img_id;
    msg.timestamp = GetSystmeTime();
    msg.index     = index;
    msg.image     = img.clone();

    inference->LoadData(msg);
  }
}

int main(int argc, char* argv[]) {
  std::string path = argv[0];
  std::size_t pos = path.find_last_of('/');
  if (pos != std::string::npos) {
    path = path.substr(0, pos);
  }

  std::string file_path_ = path + "/config/yaml/" + "yolov11_anchorfree_config.yaml";
  std::shared_ptr<InterfaceYolo> inference = InterfaceYolo::getInstance();
  inference->InitConfig(path, file_path_);
  inference->Init();
  inference->Start();

  std::string filesPath = "/home/selflearning/dataset/tinycoco/images/val2017";
  TestDemo(inference.get(), filesPath);

  inference->Stop();

  // save model result to json
  std::vector<InfertMsg> infer_msg_vec;
  inference->Callback(infer_msg_vec);
  coco_save_to_json(path + "/workspace/model_prediction.json", infer_msg_vec);

  inference->Release();
  return 0;
}
