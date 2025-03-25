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

#include "appconfig.h"

AppConfig* AppConfig::pinstance_(nullptr);
std::mutex AppConfig::mutex_;
YAML::Node AppConfig::yaml_node_;
bool AppConfig::is_init_;
bool AppConfig::quantize_flag_;
int AppConfig::src_img_w_;
int AppConfig::src_img_h_;
int AppConfig::src_img_c_;
int AppConfig::dst_img_w_;
int AppConfig::dst_img_h_;
int AppConfig::dst_img_c_;
int AppConfig::model_acc_;
int AppConfig::branch_num_;
int AppConfig::batchsizes_;
int AppConfig::infer_mode_;
int AppConfig::batch_mode_;
int AppConfig::decode_type_;
int AppConfig::max_objects_;
int AppConfig::max_batchsize_;
int AppConfig::input_msgdepth_;
int AppConfig::decode_msgdepth_;
int AppConfig::calib_batchsize_;
float AppConfig::obj_threshold_;
float AppConfig::nms_threshold_;
std::string AppConfig::model_name_;
std::string AppConfig::img_path_;
std::string AppConfig::save_img_;
std::string AppConfig::yaml_path_;
std::string AppConfig::trt_path_;
std::string AppConfig::onnx_path_;
std::string AppConfig::predict_path_;
std::string AppConfig::log_path_;
std::string AppConfig::nms_type_;
std::string AppConfig::quantize_data_;
std::string AppConfig::preprocess_type_;
std::string AppConfig::postprocess_type_;
std::string AppConfig::calib_table_path_;
std::string AppConfig::calib_preprocess_type_;

std::vector<std::vector<int>> AppConfig::predict_dim_;
std::vector<std::vector<int>> AppConfig::branchs_dim_;

YAML::Node& AppConfig::getYamlNode() { return yaml_node_; }

void AppConfig::initConfig(std::string& filename) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pinstance_ == nullptr && filename.empty()) {
    LOG(ERROR) << " There s no config file! ";
    exit(-1);
  } else if (pinstance_ == nullptr && !filename.empty()) {
    pinstance_ = new AppConfig(filename);
  }
}

AppConfig* AppConfig::getInstance() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pinstance_ == nullptr) {
    LOG(ERROR) << " Should initConfig first! ";
    exit(-1);
    return nullptr;
  }
  return pinstance_;
}

AppConfig::AppConfig(const std::string& config_filename) : config_filename_(config_filename) {
  std::ifstream in(config_filename);
  if (!in.good()) {
    LOG(ERROR) << config_filename.c_str() << " is not exist ";
    exit(-1);
  }

  yaml_path_ = config_filename_;

  yaml_node_ = YAML::LoadFile(yaml_path_);
  if (yaml_node_.IsNull()) {
    LOG(ERROR) << " Yaml config file is null ";
    return;
  }

  src_img_w_   = yaml_node_["preprocessor_config"]["src_img_width"].as<int>();
  src_img_h_   = yaml_node_["preprocessor_config"]["src_img_height"].as<int>();
  src_img_c_   = yaml_node_["preprocessor_config"]["src_img_channel"].as<int>();
  dst_img_w_   = yaml_node_["preprocessor_config"]["dst_img_width"].as<int>();
  dst_img_h_   = yaml_node_["preprocessor_config"]["dst_img_height"].as<int>();
  dst_img_c_   = yaml_node_["preprocessor_config"]["dst_img_channel"].as<int>();
  batchsizes_  = yaml_node_["preprocessor_config"]["batch_size"].as<int>();
  branch_num_  = yaml_node_["predict_config"]["branch_num"].as<int>();
  decode_type_ = yaml_node_["predict_config"]["decode_type"].as<int>();
  max_objects_ = yaml_node_["predict_config"]["max_objects"].as<int>();
  obj_threshold_  = yaml_node_["predict_config"]["obj_threshold"].as<float>();
  nms_threshold_  = yaml_node_["predict_config"]["nms_threshold"].as<float>();
  img_path_       = yaml_node_["inference_config"]["offline_test"]["img_path"].as<std::string>();
  save_img_       = yaml_node_["inference_config"]["offline_test"]["save_img"].as<std::string>();
  trt_path_       = yaml_node_["inference_config"]["engine_path"].as<std::string>();
  onnx_path_      = yaml_node_["inference_config"]["onnx_path"].as<std::string>();
  model_acc_      = yaml_node_["inference_config"]["model_acc"].as<int>();
  infer_mode_     = yaml_node_["inference_config"]["infer_mode"].as<int>();
  batch_mode_     = yaml_node_["inference_config"]["batch_mode"].as<int>();
  input_msgdepth_ = yaml_node_["inference_config"]["input_msgdepth"].as<int>();
  decode_msgdepth_= yaml_node_["inference_config"]["decode_msgdepth"].as<int>();
  predict_path_   = yaml_node_["inference_config"]["predict_path"].as<std::string>();
  log_path_       = yaml_node_["common_config"]["log_path"].as<std::string>();
  model_name_     = yaml_node_["common_config"]["model_name"].as<std::string>();
  quantize_data_  = yaml_node_["common_config"]["quantize_data"].as<std::string>();
  quantize_flag_  = yaml_node_["common_config"]["quantize_flag"].as<bool>();
  max_batchsize_  = yaml_node_["common_config"]["max_batchsize"].as<int>();
  calib_batchsize_= yaml_node_["common_config"]["calib_batchsize"].as<int>();
  calib_table_path_      = yaml_node_["common_config"]["calib_table_path"].as<std::string>();
  calib_preprocess_type_ = yaml_node_["common_config"]["calib_preprocess_type"].as<std::string>();
  nms_type_         = yaml_node_["model_config"]["nms_type"].as<std::string>();
  preprocess_type_  = yaml_node_["model_config"]["preprocess_type"].as<std::string>();
  postprocess_type_ = yaml_node_["model_config"]["postprocess_type"].as<std::string>();

  for (int index = 0; index < yaml_node_["predict_config"]["predict_dim"].size(); index++) {
    predict_dim_.push_back(yaml_node_["predict_config"]["predict_dim"][index].as<std::vector<int>>());
  }

  for (int index = 0; index < yaml_node_["predict_config"]["branchs_dim"].size(); index++) {
    branchs_dim_.push_back(yaml_node_["predict_config"]["branchs_dim"][index].as<std::vector<int>>());
  }

  if (trt_path_ == "") {
    throw std::invalid_argument("engine_path is empty");
  }
  if (onnx_path_ == "") {
    throw std::invalid_argument("onnx_path is empty");
  }
  if (predict_path_ == "") {
    throw std::invalid_argument("predict_data is empty");
  }
  if (log_path_ == "") {
    throw std::invalid_argument("log_path is empty");
  }
}
