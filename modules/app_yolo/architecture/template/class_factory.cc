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

#include "class_factory.h"
/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

RegistryFactory& RegistryFactory::getInstance() {
    static RegistryFactory instance;
    return instance;
}

std::shared_ptr<void> RegistryFactory::create(const std::string& name) {
    auto it = creators.find(name);
    if (it != creators.end()) {
        return it->second();
    }
    return { nullptr, [](void*) {} }; // 空指针的默认删除器
}

}  // namespace appinfer
}  // namespace hpc
