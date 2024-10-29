/* ==================================================================
* Copyright (c) 2024, LiWuHen.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the 
);
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

#include "appconfig.h"

AppConfig*               AppConfig::pinstance_(nullptr);
std::mutex               AppConfig::mutex_;
YAML::Node               AppConfig::yaml_node_;
bool                     AppConfig::is_init_;
int                      AppConfig::src_img_w_;
int                      AppConfig::src_img_h_;
int                      AppConfig::src_img_c_;
int                      AppConfig::dst_img_w_;
int                      AppConfig::dst_img_h_;
int                      AppConfig::dst_img_c_;
int                      AppConfig::model_acc_;
int                      AppConfig::batchsizes_;
int                      AppConfig::max_objects_;
float                    AppConfig::obj_threshold_;
float                    AppConfig::nms_threshold_;
std::string              AppConfig::img_path_;
std::string              AppConfig::save_img_;
std::string              AppConfig::yaml_path_;
std::string              AppConfig::trt_path_;
std::string              AppConfig::onnx_path_;
std::string              AppConfig::predict_path_;
std::string              AppConfig::log_path_;
std::string              AppConfig::imgs_path_;

std::vector<int>         AppConfig::predict_dim_;

YAML::Node& AppConfig::getYamlNode() {
  return yaml_node_;
}

void AppConfig::initConfig(std::string& filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(pinstance_ == nullptr && filename.empty()) {
    LOG(ERROR)<<" There s no config file! ";
    exit(-1);
  }
  else if(pinstance_ == nullptr && !filename.empty()) {
    pinstance_ = new AppConfig(filename);
  }
}

AppConfig* AppConfig::getInstance() {
  std::lock_guard<std::mutex> lock(mutex_);
  if(pinstance_ == nullptr) {
    LOG(ERROR)<<" Should initConfig first! ";
    exit(-1);
    return nullptr;
  }
  return pinstance_;
}

AppConfig::AppConfig(const std::string& config_filename) : config_filename_(config_filename) {
  std::ifstream in(config_filename);
  if(!in.good()) {
    LOG(ERROR)<<config_filename.c_str()<<" is not exist ";
    exit(-1);
  }

  yaml_path_ = config_filename_;

  yaml_node_ = YAML::LoadFile(yaml_path_);
  if(yaml_node_.IsNull()) {
    LOG(ERROR)<<" Yaml config file is null ";
    return;
  }

  src_img_w_     = yaml_node_["preprocessor_config"]["src_img_width"].as<int>();
  src_img_h_     = yaml_node_["preprocessor_config"]["src_img_height"].as<int>();
  src_img_c_     = yaml_node_["preprocessor_config"]["src_img_channel"].as<int>();
  dst_img_w_     = yaml_node_["preprocessor_config"]["dst_img_width"].as<int>();  
  dst_img_h_     = yaml_node_["preprocessor_config"]["dst_img_height"].as<int>(); 
  dst_img_c_     = yaml_node_["preprocessor_config"]["dst_img_channel"].as<int>();
  batchsizes_    = yaml_node_["preprocessor_config"]["batch_size"].as<int>();
  predict_dim_   = yaml_node_["predict_config"]["predict_dim"].as<std::vector<int>>();
  max_objects_   = yaml_node_["predict_config"]["max_objects"].as<int>();
  obj_threshold_ = yaml_node_["predict_config"]["obj_threshold"].as<float>();
  nms_threshold_ = yaml_node_["predict_config"]["nms_threshold"].as<float>();
  img_path_      = yaml_node_["inference_config"]["offline_test"]["img_path"].as<std::string>();
  save_img_      = yaml_node_["inference_config"]["offline_test"]["save_img"].as<std::string>();
  trt_path_      = yaml_node_["inference_config"]["engine_path"].as<std::string>();
  onnx_path_     = yaml_node_["inference_config"]["onnx_path"].as<std::string>();
  model_acc_     = yaml_node_["inference_config"]["model_acc"].as<int>();
  predict_path_  = yaml_node_["inference_config"]["predict_path"].as<std::string>();
  log_path_      = yaml_node_["common_config"]["log_path"].as<std::string>();
  imgs_path_     = yaml_node_["common_config"]["imgs_path"].as<std::string>();

  if(trt_path_ == "") {
    throw std::invalid_argument("engine_path is empty");
  }
  if(onnx_path_ == "") {
    throw std::invalid_argument("onnx_path is empty");
  }
  if(predict_path_ == "") {
    throw std::invalid_argument("predict_data is empty");
  }
  if(log_path_ == "") {
    throw std::invalid_argument("log_path is empty");
  }

}
