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

#include "parseconfig.h"

namespace hpc {
namespace common {

ParseMsgs::ParseMsgs() { ReadYamlParam(); }

ParseMsgs::~ParseMsgs() {}

/**
 * @brief Read yaml file.
 */
void ParseMsgs::ReadYamlParam() {
  auto app_config = AppConfig::getInstance();
  yaml_path_ = app_config->get_yaml_path_();
  src_img_w_ = app_config->get_src_img_w_();
  src_img_h_ = app_config->get_src_img_h_();
  src_img_c_ = app_config->get_src_img_c_();
  dst_img_w_ = app_config->get_dst_img_w_();
  dst_img_h_ = app_config->get_dst_img_h_();
  dst_img_c_ = app_config->get_dst_img_c_();
  model_acc_ = app_config->get_model_acc_();
  nms_type_  = app_config->get_nms_type_();
  model_name_     = app_config->get_model_name_();
  infer_mode_     = app_config->get_infer_mode_();
  batch_mode_     = app_config->get_batch_mode_();
  branch_num_     = app_config->get_branch_num_();
  batchsizes_     = app_config->get_batchsizes_();
  branchs_dim_    = app_config->get_branchs_dim_();
  predict_dim_    = app_config->get_predict_dim_();
  decode_type_    = app_config->get_decode_type_();
  max_objects_    = app_config->get_max_objects_();
  quantize_data_  = app_config->get_quantize_data_();
  quantize_flag_  = app_config->get_quantize_flag_();
  obj_threshold_  = app_config->get_obj_threshold_();
  nms_threshold_  = app_config->get_nms_threshold_();
  max_batchsize_  = app_config->get_max_batchsize_();
  input_msgdepth_ = app_config->get_input_msgdepth_();
  calib_batchsize_= app_config->get_calib_batchsize_();
  decode_msgdepth_= app_config->get_decode_msgdepth_();
  preprocess_type_= app_config->get_preprocess_type_();
  img_path_       = app_config->get_home_path_() + app_config->get_img_path_();
  save_img_       = app_config->get_home_path_() + app_config->get_save_img_();
  trt_path_       = app_config->get_home_path_() + app_config->get_trt_path_();
  onnx_path_      = app_config->get_home_path_() + app_config->get_onnx_path_();
  predict_path_   = app_config->get_home_path_() + app_config->get_predict_path_();
  log_path_       = app_config->get_home_path_() + app_config->get_log_path_();

  postprocess_type_      = app_config->get_postprocess_type_();
  calib_table_path_      = app_config->get_calib_table_path_();
  calib_preprocess_type_ = app_config->get_calib_preprocess_type_();

  srcimg_size_   = src_img_w_ * src_img_h_ * src_img_c_;
  dstimg_size_   = dst_img_w_ * dst_img_h_ * dst_img_c_;
}

}  // namespace common
}  // namespace hpc
