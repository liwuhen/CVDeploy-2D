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

#ifndef APP_YOLO_TEMPLATE_FACTORY_H__
#define APP_YOLO_TEMPLATE_FACTORY_H__

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

// 内部注册表
class RegistryFactory {
public:
    static RegistryFactory& getInstance();
    template<typename T>
    void registerCreator(const std::string& name) {
        creators[name] = []() { return std::shared_ptr<T>(new T()); };
    }
    std::shared_ptr<void> create(const std::string& name);
private:
    RegistryFactory() = default;
    std::unordered_map<std::string, std::function<std::shared_ptr<void>()>> creators;
};

// 用户创建对象的泛型接口
template<typename T>
std::shared_ptr<T> createObject(const std::string& name) {
    auto ptr = RegistryFactory::getInstance().create(name);
    if (!ptr) return nullptr;
    // 假设用户知道类型匹配，否则会导致未定义行为
    return std::static_pointer_cast<T>(ptr);
}

// 辅助宏，用于简化用户注册
#define REGISTER_CLASS(NAME, CLASS) \
    namespace { \
        struct RegistryFactory_##CLASS { \
            RegistryFactory_##CLASS() { \
                RegistryFactory::getInstance().registerCreator<CLASS>(NAME); \
            } \
        } registrarfactory_##CLASS; \
    }

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_TEMPLATE_FACTORY_H__
