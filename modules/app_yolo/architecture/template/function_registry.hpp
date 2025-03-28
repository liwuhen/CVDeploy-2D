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

#ifndef APP_YOLO_QUANTIZE_CALIBREGISTRY_H__
#define APP_YOLO_QUANTIZE_CALIBREGISTRY_H__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

// 注册器类
class Registry {
public:
    static std::shared_ptr<Registry> getInstance() {
        static std::shared_ptr<Registry> instance(new Registry());
        return instance;
    }

    // 注册函数，接受任意 std::function
    void registerFunc(const std::string& name, std::shared_ptr<void> cb) {
        callbacks[name] = cb;
    }

    template<typename... Args>
    std::function<void(Args...)> getRegisterFunc(const std::string& name) {
        auto it = callbacks.find(name);
        if (it != callbacks.end()) {
            auto holder = std::static_pointer_cast<std::function<void(Args...)>>(it->second);
            if (holder) {
                return *holder;
            }
        }
        return nullptr;  // 未找到返回 nullptr
    }

private:
    Registry() = default;
    std::unordered_map<std::string, std::shared_ptr<void>> callbacks;
};

// 宏定义，使用运行时注册
#define REGISTER_CALIBRATOR_FUNC(name, func) \
    namespace { \
        struct AutoReg_##func { \
            AutoReg_##func() { \
                auto cb = std::make_shared<std::function<decltype(func)>>(func); \
                Registry::getInstance()->registerFunc(name, cb); \
            } \
        }; \
        static AutoReg_##func reg_##func; \
    }

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_YOLO_QUANTIZE_CALIBREGISTRY_H__
