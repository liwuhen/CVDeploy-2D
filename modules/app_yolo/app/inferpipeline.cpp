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

#include "inferpipeline.h"

namespace hpc{

namespace appinfer{

InferPipeline::InferPipeline() {}

InferPipeline::~InferPipeline() {}

/**
 * @description: init．
*/
bool InferPipeline::Init()
{
    InitLogging();
    
    moduleMap_[ModuleType::MODULE_INFERENCE]   = std::make_shared<InferenceEngine>();
    moduleMap_[ModuleType::MODULE_POSTPROCESS] = std::make_shared<PostProcessor>();

    this->parsemsgs_ = std::make_shared<ParseMsgs>();

    for ( auto &module : moduleMap_)  {
        if ( !module.second->SetParam(parsemsgs_) ) {
            GLOG_ERROR(module.first<<" Module set param failed ");
            return false;
        }
    }

    for ( auto &module : moduleMap_)  {
        if ( !module.second->Init() ) {
            GLOG_ERROR(module.first<<" Module init failed ");
            return false;
        }
    }

    std::map<ModuleType, std::vector<ModuleType>> modules_map = {
        {ModuleType::MODULE_POSTPROCESS, {
           ModuleType::MODULE_INFERENCE}}
    };

    for ( auto& [module_type,  module_vec] : modules_map ) {
        for ( auto& pre_module : module_vec ) {
            if ( !moduleMap_[module_type]->SetModulePipeline(pre_module, moduleMap_[pre_module])) {
                return false;
            }
        }
    }

    GLOG_INFO("[Init]: ALL module init success ");
    return true;
}

/**
 * @description: The inference algorithm handles threads．
*/
bool InferPipeline::Start()
{
    for ( auto &module : moduleMap_)  {
        if ( !module.second->RunStart() ) {
            GLOG_ERROR(module.first<<" Module start failed ");
            return false;
        };
    }

    GLOG_INFO("[Start]: ALL module start ");
    return true;
}
        
/**
 * @description: Thread stop． 
*/
bool InferPipeline::Stop()
{
    for ( auto &module : moduleMap_)  {
        if ( !module.second->RunStop() ) {
            GLOG_ERROR(module.first<<" Module stop failed ");
            return false;
        };
    }

    GLOG_INFO("[Stop]: ALL module stop ");
    return true;
}

/**
 * @description: Software function stops． 
*/
bool InferPipeline::Release()
{
    for ( auto &module : moduleMap_)  {
        if ( !module.second->RunRelease() ) {
            GLOG_ERROR(module.first<<" Module release failed ");
            return false;
        };
    }

    GLOG_INFO("[Release]: ALL module release ");
    return true;
}

/**
 * @description: Load data.
*/
bool InferPipeline::LoadData(InfertMsg& infer_msg) 
{
    if ( moduleMap_.find(ModuleType::MODULE_INFERENCE) != moduleMap_.end() ) {
        auto inference = dynamic_cast<InferenceEngine *>(moduleMap_[ModuleType::MODULE_INFERENCE].get());
        inference->LoadData(infer_msg);
    } else {
        GLOG_ERROR("Inference module does not exit!");
    }
    return true;
}

/**
 * @description: Module monitoring. 
*/
bool InferPipeline::ModuleMonitor()
{}

} //namespace appinfer
} //namespace hpc