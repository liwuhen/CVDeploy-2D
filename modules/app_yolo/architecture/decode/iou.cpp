#include "iou.h"

namespace hpc {
namespace appinfer {

IouPlugin::IouPlugin() {}

IouPlugin::~IouPlugin() {}

/**
 * @description: init．
 */
bool IouPlugin::Init() {
  GLOG_INFO("[Init]: IouPlugin module init ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool IouPlugin::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: IouPlugin module set param failed ");
    return false;
  }

  GLOG_INFO("[SetParam]: IouPlugin module set param ");
  return true;
}

/**
 * @description: Iou.
 */
float IouPlugin::Iou(Box& box_a, Box& box_b) {
  // iou 计算
  float iou_result = 0.0f;
  float cross_left = std::max(box_a.left, box_b.left);
  float cross_top = std::max(box_a.top, box_b.top);
  float cross_right = std::min(box_a.right, box_b.right);
  float cross_bottom = std::min(box_a.bottom, box_b.bottom);

  float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
  float union_area =
      std::max(0.0f, box_a.right - box_a.left) * std::max(0.0f, box_a.bottom - box_a.top) + std::max(0.0f, box_b.right - box_b.left) * std::max(0.0f, box_b.bottom - box_b.top) - cross_area;
  if (cross_area == 0 || union_area == 0) {
    iou_result = 0.0f;
  }
  iou_result = cross_area / union_area;

  return iou_result;
}

}  // namespace appinfer
}  // namespace hpc
