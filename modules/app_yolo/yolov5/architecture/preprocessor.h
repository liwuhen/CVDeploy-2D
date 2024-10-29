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

#ifndef APP_YOLO_PREPROCESSOR_V5_H__
#define APP_YOLO_PREPROCESSOR_V5_H__

#include <iostream>
#include <time.h>
#include <map>
#include <chrono>
#include <fstream>
#include <vector>
#include <mutex>
#include <future>
#include <functional>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "module.h"
#include "utils.hpp"
#include "common.hpp"
#include "enum_msg.h"
#include "parseconfig.h"
#include "warpaffine.hpp"
#include "task_struct.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc{
namespace appinfer{

using namespace std;
using namespace Eigen;
using namespace hpc::common;

/**
 * @class PreProcessor.
 * @brief Model preprocessing.
 */
class PreProcessor : public InferModuleBase {
public:
    PreProcessor();
    ~PreProcessor();

    /**
     * @brief     init．
     * @param[in] void．
     * @return    bool.
     */
    bool Init() override;

    /**
     * @brief     The inference algorithm handles threads．
     * @param[in] void．
     * @return    bool.
     */
    bool RunStart() override;

    /**
     * @brief     Thread stop．
     * @param[in] void．
     * @return    bool.
     */
    bool RunStop() override;

    /**
     * @brief     Software function stops．
     * @param[in] void．
     * @return    bool.
     */
    bool RunRelease() override;

    /**
     * @brief     Configuration parameters.
     * @param[in] shared_ptr<ParseMsgs>&.
     * @return    bool.
     */
    bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) override;

    /**
     * @brief     Cpu and gpu memory free.
     * @param[in] void.
     * @return    bool.
     */
    bool MemFree();

    /**
     * @brief     Inference.
     * @param[in] float*.
     * @return    bool.
     */
    bool Inference(InfertMsg& input_msg, 
            int batch, float * dstimg, DeviceMode inferMode, cudaStream_t stream); 

private: 
    
    /**
     * @brief     Module resource release.
     * @param[in] void．
     * @return    bool.
     */
    bool DataResourceRelease();

    /**
     * @brief     Memory allocator.
     * @param[in] void．
     * @return    bool.
     */
    bool MemAllocator();

    /**
     * @brief     cal affine matrix.
     * @param[in] InfertMsg&
     * @return    void.
     */
    void CalAffineMatrix(InfertMsg& input_msg);

    /**
     * @brief     Gpu preprocessor.
     * @param[in] [InfertMsg&, int, float *, cudaStream_t].
     * @return    bool.
     */
    bool GpuPreprocessor(InfertMsg& input_msg, 
            int batch, float * dstimg, cudaStream_t stream);

    /**
     * @brief     Cpu preprocessor.
     * @param[in] [cv::Mat&, float *, cudaStream_t].
     * @return    bool.
     */
    bool CpuPreprocessor(cv::Mat& srcimg, 
            uint64_t timestamp, float * input_device_gpu, cudaStream_t stream);


private:

    float* input_data_host_;  
    uint8_t* input_data_device_;  

    std::shared_ptr<ParseMsgs> parsemsgs_;

};

} //namespace appinfer
} //namespace hpc

#endif // APP_YOLO_PREPROCESSOR_V5_H__