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

#ifndef APP_AUTOPILOT_UTILS_H__
#define APP_AUTOPILOT_UTILS_H__

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;

typedef std::function<void(const float*)> CallbackFun;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

static bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op,
           err_name, err_message);
    return false;
  }
  return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
  switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
      return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
      return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
      return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
      return "verbose";
    default:
      return "unknown";
  }
}

class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if (severity <= Severity::kINFO) {
      // 打印带颜色的字符，格式如下：
      // printf("\033[47;33m 打印的文本\033[0m");
      // 其中 \033[ 是起始标记
      //      47    是背景颜色
      //      ;     分隔符
      //      33    文字颜色
      //      m     开始标记结束
      //      \033[0m 是终止标记
      // 其中背景颜色或者文字颜色可不写
      // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
      if (severity == Severity::kWARNING) {
        printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
      } else if (severity <= Severity::kERROR) {
        printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
      } else {
        printf("%s: %s\n", severity_string(severity), msg);
      }
    }
  }
};

struct NvInferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    // if (obj){
    // obj->destroy();
    // }
  }
};

// 通过智能指针管理 nv 返回的指针参数
// 内存自动释放，避免泄漏
template <typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr) {
  return std::shared_ptr<_T>(ptr, [](_T* p) { p->destroy(); });
}

static string join_dims(const vector<int>& dims){
  stringstream output;
  char buf[64];
  const char* fmts[] = {"%d", " x %d"};
  for(int i = 0; i < dims.size(); ++i){
    snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
    output << buf;
  }
  return output.str();
}


}  // namespace appinfer
}  // namespace hpc

#endif  // APP_AUTOPILOT_UTILS_H__
