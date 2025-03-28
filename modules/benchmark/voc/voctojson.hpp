#include <iostream>
#include "json.hpp"
#include "file.hpp"
#include "task_struct.hpp"

using namespace std;
using namespace hpc::common;

static bool voc_save_to_json(const string& file,
    std::vector<InfertMsg>& infer_msg_vec){

    Json::Value predictions(Json::arrayValue);
    for( auto& msg : infer_msg_vec ) {
        uint32_t image_id = msg.frame_id;

        auto& boxes = msg.bboxes;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"]    = image_id;
            jitem["category_id"] = box.label;
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
