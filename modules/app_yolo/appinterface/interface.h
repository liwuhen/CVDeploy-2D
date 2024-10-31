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

#ifndef APP_YOLO_INTERFACE_H__
#define APP_YOLO_INTERFACE_H__

#include <iostream>
#include <memory>

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc{
namespace appinfer{

using namespace std;

/**
 * @class InterfaceYolo.
 * @brief Interface.
 */
class InterfaceYolo {

public:
    
    static std::shared_ptr<InterfaceYolo> getInstance() {
        static std::shared_ptr<InterfaceYolo> instance(new InterfaceYolo());
        return instance;
    }

    ~InterfaceYolo() {};

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
     * @param[in] T&.
     * @return    bool.
     */
    template <typename T>
    bool LoadData(T& class_t);

    /**
     * @brief     Init config.
     * @param[in] template <typename... Args.
     * @return    bool.
     */
    template <typename... Args>
    bool InitConfig(Args&&... args);

private:
    InterfaceYolo() = default;
    InterfaceYolo(const InterfaceYolo&) = delete;
    InterfaceYolo& operator=(const InterfaceYolo&) = delete;

private:
    class InferInterFace;
    std::shared_ptr<InferInterFace>infer_;
};

} //namespace appinfer
} //namespace hpc

#endif // APP_YOLO_INTERFACE_H__