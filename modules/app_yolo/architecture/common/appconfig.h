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

#ifndef APP_YOLO_APPCONFIG_H__
#define APP_YOLO_APPCONFIG_H__

#include <glog/logging.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "colorprintf.h"
#include "common.hpp"
#include "yaml-cpp/yaml.h"

#define REG_YAML_PUBLIC_VAR(type, name) \
 public:                                \
  static type name;                     \
                                        \
 public:                                \
  static type& get_##name() { return name; }

#define REG_YAML_VAR(type, name) \
 private:                        \
  static type name;              \
                                 \
 public:                         \
  static type& get_##name() { return name; }

#define REG_CUS_VAR(name)                          \
 private:                                          \
  std::string name;                                \
                                                   \
 public:                                           \
  void set_##name(std::string var) { name = var; } \
  std::string get_##name() { return name; }

class AppConfig {
 private:
  static AppConfig* pinstance_;
  static std::mutex mutex_;
  static YAML::Node yaml_node_;

  REG_CUS_VAR(home_path_);
  REG_YAML_VAR(bool, is_init_);
  REG_YAML_VAR(bool, quantize_flag_);
  REG_YAML_VAR(int, src_img_w_);
  REG_YAML_VAR(int, src_img_h_);
  REG_YAML_VAR(int, src_img_c_);
  REG_YAML_VAR(int, dst_img_w_);
  REG_YAML_VAR(int, dst_img_h_);
  REG_YAML_VAR(int, dst_img_c_);
  REG_YAML_VAR(int, branch_num_);
  REG_YAML_VAR(int, batchsizes_);
  REG_YAML_VAR(int, decode_type_);
  REG_YAML_VAR(int, max_objects_);
  REG_YAML_VAR(int, model_acc_);
  REG_YAML_VAR(int, infer_mode_);
  REG_YAML_VAR(int, batch_mode_);
  REG_YAML_VAR(int, max_batchsize_);
  REG_YAML_VAR(int, calib_batchsize_);
  REG_YAML_VAR(int, input_msgdepth_);
  REG_YAML_VAR(int, decode_msgdepth_);
  REG_YAML_VAR(float, obj_threshold_);
  REG_YAML_VAR(float, nms_threshold_);
  REG_YAML_VAR(std::string, nms_type_);
  REG_YAML_VAR(std::string, img_path_);
  REG_YAML_VAR(std::string, save_img_);
  REG_YAML_VAR(std::string, yaml_path_);
  REG_YAML_VAR(std::string, trt_path_);
  REG_YAML_VAR(std::string, onnx_path_);
  REG_YAML_VAR(std::string, predict_path_);
  REG_YAML_VAR(std::string, log_path_);
  REG_YAML_VAR(std::string, model_name_);
  REG_YAML_VAR(std::string, quantize_data_);
  REG_YAML_VAR(std::string, preprocess_type_);
  REG_YAML_VAR(std::string, postprocess_type_);
  REG_YAML_VAR(std::string, calib_table_path_);
  REG_YAML_VAR(std::string, calib_preprocess_type_);


  REG_YAML_VAR(std::vector<std::vector<int>>, predict_dim_);
  REG_YAML_VAR(std::vector<std::vector<int>>, branchs_dim_);

 protected:
  explicit AppConfig(const std::string& config_filename);
  ~AppConfig() {}
  std::string config_filename_;

 public:
  AppConfig(AppConfig&) = delete;
  void operator=(const AppConfig&) = delete;
  static AppConfig* getInstance();
  static void initConfig(std::string& filename);
  static YAML::Node& getYamlNode();
};

inline bool InitAppConfig(std::string& homepath, std::string& filepath) {
  AppConfig::initConfig(filepath);
  AppConfig::getInstance()->set_home_path_(homepath);
  return true;
}

#endif  // APP_YOLO_APPCONFIG_H__
