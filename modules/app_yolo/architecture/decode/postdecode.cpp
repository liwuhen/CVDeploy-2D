#include "postdecode.h"

namespace hpc {
namespace appinfer {

/**
 * @description: init．
 */
bool ModelDecode::Init() {
  // nms_plugin_ = std::make_shared<NmsPlugin>();
  // nms_plugin_->SetParam(parsemsgs_);
  // nms_plugin_->Init();

  feat_sizes_ = {{64, 120}, {32, 60}, {16, 30}};

  anchor_points_ = Generate_Anchor_Points();

  GLOG_INFO("[Init]: ModelV5Decode module init ");
  return true;
}

/**
 * @brief The inference algorithm handles threads．
 */
bool ModelDecode::RunStart() {
  GLOG_INFO("[RunStart]: ModelDecode module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool ModelDecode::RunStop() {
  GLOG_INFO("[RunStop]: ModelDecode module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool ModelDecode::RunRelease() {
  GLOG_INFO("[RunRelease]: ModelDecode module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool ModelDecode::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: ModelDecode module set param failed ");
    return false;
  }
  imgshape_["dst"] = make_pair(parsemsgs_->dst_img_h_, parsemsgs_->dst_img_w_);

  GLOG_INFO("[SetParam]: ModelDecode module set param ");
  return true;
}

/**
 * @description: Cal anchor.
 */
AnchorPointsVector ModelDecode::Generate_Anchor_Points() {
    AnchorPointsVector anchor_points;
    for (int i = 0; i < 3; i++) {
        std::vector<std::pair<int, int>> anchors;
        int feat_size = feat_sizes_[i].first * feat_sizes_[i].second;
        for (int j = 0; j < feat_size; j++) {
            int grid_x = j % feat_sizes_[i].second;
            int grid_y = j / feat_sizes_[i].second;
            anchors.push_back(std::make_pair(grid_x, grid_y));
        }
        anchor_points.push_back(anchors);
    }

    return anchor_points;
}

/**
 * @description: Bounding box decoding at feature level．
 */
void ModelDecode::BboxDecodeFeatureLevel(std::vector<float*>& predict,
    InfertMsg& infer_msg, vector<Box>& box_result) {
  int label     = 0;
  float prob    = 0.0f;
  float objness = 0.0f;
  int stride    = 0;
  float grid_x, grid_y = 0.0f;
  float cx, cy, width, height = 0.0f;

  vector<Box> boxes;

  int l_size = parsemsgs_->branchs_dim_[0][1] * parsemsgs_->branchs_dim_[0][2] * parsemsgs_->branchs_dim_[0][3];
  int d_size = parsemsgs_->branchs_dim_[1][1] * parsemsgs_->branchs_dim_[1][2] * parsemsgs_->branchs_dim_[1][3];
  int s_size = parsemsgs_->branchs_dim_[2][1] * parsemsgs_->branchs_dim_[2][2] * parsemsgs_->branchs_dim_[2][3];

  int predict_outs = parsemsgs_->predict_dim_[0][1];
  for (int i = 0; i < predict_outs; ++i)
  {
    // cal anchor point
    if (i < l_size) {
        grid_x = anchor_points_[0][i].first;
        grid_y = anchor_points_[0][i].second;
        stride = 8;
    } else if (i >= l_size && i < l_size + d_size) {
        grid_x = anchor_points_[1][i-l_size].first;
        grid_y = anchor_points_[1][i-l_size].second;
        stride = 16;
    } else if (i >= l_size + d_size && i < l_size + d_size + s_size) {
        grid_x = anchor_points_[2][i-l_size-d_size].first;
        grid_y = anchor_points_[2][i-l_size-d_size].second;
        stride = 32;
    }

    std::vector<float*> outvec;
    int label_num = parsemsgs_->predict_dim_[0][2] - 5;
    if (parsemsgs_->branch_num_ == (int)DecodeBranch::FEATURE_THREE) {
      for (int j = 0; j < parsemsgs_->branch_num_; j++) {
        outvec.push_back(predict[j] + i * parsemsgs_->predict_dim_[j][2]);  // cls score boxes 特征图级别
      }

      label   = std::max_element(outvec[0], outvec[0] + label_num) - outvec[0];
      prob    = outvec[0][label];
      objness = outvec[1][0];

      // 特征图级别 -> 输入图像层级
      cx     = (outvec[2][0] + grid_x) * stride;  // 输入图像级别
      cy     = (outvec[2][1] + grid_y) * stride;
      width  = exp(outvec[2][2]) * stride;
      height = exp(outvec[2][3]) * stride;  // anchor free

    } else if (parsemsgs_->branch_num_ == (int)DecodeBranch::FEATURE_ONE) {
      outvec.push_back(predict[0] + i * parsemsgs_->predict_dim_[0][2]);  // boxes infos 特征图级别

      float* lable_score = outvec[0] + 5;
      label   = std::max_element(lable_score, lable_score + label_num) - lable_score;
      prob    = lable_score[label];
      objness = outvec[0][5];

      // 特征图级别 -> 输入图像层级
      cx     = (outvec[0][0] + grid_x) * stride;  // 输入图像级别
      cy     = (outvec[0][1] + grid_y) * stride;
      width  = exp(outvec[0][2]) * stride;
      height = exp(outvec[0][3]) * stride;  // anchor free
    }

    float confidence = prob * objness;
    if(confidence < parsemsgs_->obj_threshold_)
        continue;

    // 输入图像级别
    float left   = cx - width  * 0.5;
    float top    = cy - height * 0.5;
    float right  = cx + width  * 0.5;
    float bottom = cy + height * 0.5;

    // 输入图像层级模型预测框 ==> 映射回原图上尺寸
    float image_left   = infer_msg.affineMatrix_inv(0, 0) * left   + infer_msg.affineMatrix_inv(0, 2);
    float image_top    = infer_msg.affineMatrix_inv(1, 1) * top    + infer_msg.affineMatrix_inv(1, 2);
    float image_right  = infer_msg.affineMatrix_inv(0, 0) * right  + infer_msg.affineMatrix_inv(0, 2);
    float image_bottom = infer_msg.affineMatrix_inv(1, 1) * bottom + infer_msg.affineMatrix_inv(1, 2);
    boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
  }
  // nms_plugin_->Nms(boxes, box_result, parsemsgs_->nms_threshold_);
}

/**
 * @description: Bounding box decoding at input level．
 */
void ModelDecode::BboxDecodeInputLevel(std::vector<float*>& predict,
    InfertMsg& infer_msg, vector<Box>& box_result) {

  vector<Box> boxes;
  int num_classes = parsemsgs_->predict_dim_[0][2] - 5;
  for (int i = 0; i < parsemsgs_->predict_dim_[0][1]; ++i)
  {
    float* pitem  = predict[0] + i * parsemsgs_->predict_dim_[0][2];
    float* pclass = pitem + 5;

    float objitem = pitem[4];
    if ( objitem < parsemsgs_->obj_threshold_ ) continue;

    int label  = std::max_element(pclass, pclass + num_classes) - pclass;
    float prob = pclass[label];
    float confidence = prob * objitem;    // anchor free
    if (confidence < parsemsgs_->obj_threshold_) continue;

    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];
    float left   = cx - width  * 0.5;
    float top    = cy - height * 0.5;
    float right  = cx + width  * 0.5;
    float bottom = cy + height * 0.5;

    // 输入图像层级模型预测框 ==> 映射回原图上尺寸
    float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                         + infer_msg.affineMatrix_inv(0, 2);
    float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                         + infer_msg.affineMatrix_inv(1, 2);
    float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                         + infer_msg.affineMatrix_inv(0, 2);
    float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                         + infer_msg.affineMatrix_inv(1, 2);

    if ( image_left < 0 || image_top< 0 ) {
      continue;
    }

    boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
  }

  // nms_plugin_->Nms(boxes, box_result, parsemsgs_->nms_threshold_);
}

}  // namespace appinfer
}  // namespace hpc
