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

#ifndef APP_YOLO_MODULE_H__
#define APP_YOLO_MODULE_H__

#include <iostream>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "parseconfig.h"
#include "module_struct.h"
/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc{
namespace appinfer{

using namespace std;
using namespace hpc::common;
/**
 * @class InferModuleBase.
 * @brief Model inference base.
 */
class InferModuleBase {
public:
    InferModuleBase() {};
    virtual ~InferModuleBase() = default;

    virtual bool Init() = 0;

    virtual bool RunStart() = 0;

    virtual bool RunStop() = 0;

    virtual bool RunRelease() = 0;

    virtual bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) = 0;

public:
    bool SetModulePipeline(ModuleType module_type, std::shared_ptr<InferModuleBase> &module) {
        if ( (!moduleSets_.count(module_type)) ) {
            moduleSets_[module_type] = module;
            return true;
        }
        return false;
    }

    std::shared_ptr<InferModuleBase> GetModulePipeline(ModuleType module_type) {
        if ( moduleSets_.count(module_type) ) {
            return moduleSets_[module_type];
        } 
        return nullptr;
    }

private:
    std::unordered_map<ModuleType, std::shared_ptr<InferModuleBase>> moduleSets_;

};

} //namespace appinfer
} //namespace hpc

#endif // APP_YOLO_MODULE_H__