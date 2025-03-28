#include <iostream>
#include "json.hpp"
#include "file.hpp"
#include "task_struct.hpp"

using namespace std;
using namespace hpc::common;

static bool coco_save_to_json(const string& file,
    std::vector<InfertMsg>& infer_msg_vec){

    int to_coco90_class_map[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    };
    Json::Value predictions(Json::arrayValue);
    for( auto& msg : infer_msg_vec ) {
        uint32_t image_id = msg.frame_id;

        auto& boxes = msg.bboxes;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"]    = image_id;
            jitem["category_id"] = to_coco90_class_map[box.label];
            jitem["score"]       = box.confidence;

            auto& bbox = jitem["bbox"];
            bbox.append(box.left);
            bbox.append(box.top);
            bbox.append(box.right - box.left);
            bbox.append(box.bottom - box.top);
            predictions.append(jitem);
        }
    }
    return FileSystem::save_file(file, predictions.toStyledString());
}
