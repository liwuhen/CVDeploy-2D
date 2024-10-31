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

#ifndef APP_YOLO_INFER_PIPELINE_H__
#define APP_YOLO_INFER_PIPELINE_H__

#include <iostream>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include "logger.h"
#include "glog_msg.h"
#include "common.hpp"
#include "inference.h"
#include "parseconfig.h"
#include "postprocessor.h"
#include "module_struct.h"
#include "task_struct.hpp"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc{
namespace appinfer{

using namespace std;
using namespace hpc::common;

/**
 * @class InferPipeline.
 * @brief Model inference.
 */
class InferPipeline {
public:
    InferPipeline();
    ~InferPipeline();

    /**
     * @brief     init．
     * @param[in] void．
     * @return    bool.
     */
    bool Init();

    /**
     * @brief     The inference algorithm handles threads．
     * @param[in] void．
     * @return    bool.
     */
    bool Start();

    /**
     * @brief     Thread stop．
     * @param[in] void．
     * @return    bool.
     */
    bool Stop();

    /**
     * @brief     Software function stops．
     * @param[in] void．
     * @return    bool.
     */
    bool Release();

    /**
     * @brief     Load data.
     * @param[in] InfertMsg&.
     * @return    bool.
     */
    bool LoadData(InfertMsg& infer_msg);

private:
    /**
     * @brief     Module monitoring.
     * @param[in] void．
     * @return    bool.
     */
    bool ModuleMonitor();

private:  
    std::vector<std::string> param_path_;
    std::shared_ptr<ParseMsgs> parsemsgs_;
    std::map<ModuleType, std::shared_ptr<InferModuleBase>> moduleMap_;

};

} //namespace appinfer
} //namespace hpc

#endif // APP_YOLO_INFER_PIPELINE_H__