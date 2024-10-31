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

#ifndef APP_COMMON_IMAGE_MSG_H__
#define APP_COMMON_IMAGE_MSG_H__

#include <iostream>
#include <mutex>
#include <memory>
#include <queue>
#include <string>
#include <list>
#include <functional>
#include <opencv2/opencv.hpp>
#include "std_buffer.h"


namespace hpc {

namespace common {

class ImageInfos{
public:
    ImageInfos() {
        width_     = 0;
        height_    = 0;
        frame_id_  = 0;
        timestamp_ = 0;
        image_     = cv::Mat();
    };
    ~ImageInfos() {};

    bool DeepCopy(const ImageInfos *src) {
        this->image_ = src->image_.clone();
        return true;
    };

    bool MoveCopy(const ImageInfos *src) {
        this->image_ = std::move(src->image_);
        return true;
    };

public:
    uint16_t width_;
    uint16_t height_;
    uint32_t frame_id_;
    int64_t timestamp_;
    cv::Mat image_;
};

typedef std::shared_ptr<ImageInfos> image_ptr;

}
}

#endif // APP_COMMON_IMAGE_MSG_H__
