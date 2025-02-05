#include "nms.h"

namespace hpc {
namespace appinfer {

NmsPlugin::NmsPlugin() {}

NmsPlugin::~NmsPlugin() {}

/**
 * @description: init．
 */
bool NmsPlugin::Init() {
  iou_plugin_ = std::make_shared<IouPlugin>();
  iou_plugin_->SetParam(parsemsgs_);
  iou_plugin_->Init();

  GLOG_INFO("[Init]: NmsPlugin module init ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool NmsPlugin::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: NmsPlugin module set param failed ");
    return false;
  }

  GLOG_INFO("[SetParam]: NmsPlugin module set param ");
  return true;
}

/**
 * @description: Nms.
 */
bool NmsPlugin::Nms(vector<Box>& boxes, vector<Box>& box_result, float nms_threshold) {
  // nms 过程：
  // 1、按照类别概率排序，2、从最大概率矩形框开始，与其他框判断 iou
  // 是否超过阈值，3、标记重叠度超过阈值的框，丢掉，4、从剩余框中选择概率最大框并于剩余框判断重叠度是否超过阈值，重复步骤
  // 3 过程
  std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b) { return a.confidence > b.confidence; });  // 这里给的引用，可以避免拷贝。从大到小排序
  std::vector<bool> remove_flags(boxes.size());                                                       // 初始化容器容量，并预先分配内存。
  box_result.reserve(boxes.size());                                                                   // reserve 主动分配内存可以提升程序执行效率

  for (int i = 0; i < boxes.size(); ++i) {
    if (remove_flags[i]) continue;

    auto& ibox = boxes[i];
    box_result.emplace_back(ibox);
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (remove_flags[j]) continue;

      auto& jbox = boxes[j];
      if (ibox.label == jbox.label) {
        // class matched
        if (iou_plugin_->Iou(ibox, jbox) >= nms_threshold) remove_flags[j] = true;
      }
    }
  }

  return true;
}

}  // namespace appinfer
}  // namespace hpc
