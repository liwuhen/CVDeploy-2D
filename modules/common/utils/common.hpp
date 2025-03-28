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

#ifndef APP_COMMON_COMMON_H__
#define APP_COMMON_COMMON_H__

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "colorprintf.h"
#include "enum_msg.h"

/**
 * @namespace hpc::common
 * @brief hpc::common
 */
namespace hpc {
namespace common {

using namespace std;

/**
 * @description: Determine if a file exists in the specified path.
 */
static bool isFileExists_stat(string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

/**
 * @description: Saving a serialisation file.
 */
static bool save_file(const std::string& file, const void* data, size_t length) {
  FILE* f = fopen(file.c_str(), "wb");
  if (!f) {
    return false;
  }

  if (data && length > 0) {
    if (fwrite(data, 1, length, f) != length) {
      fclose(f);
      return false;
    }
  }
  fclose(f);
  return true;
}

/**
 * @description: Get now time.
 */
static double timestamp_now_float() {
  return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}

/**
 * @description: Get average now time.
 */
static std::vector<double> GetAverageTime(std::vector<double>& preproccess_time,
  std::vector<double>& infer_time,
  std::vector<double>& decode_time,
  std::vector<double>& endtoend_time) {

    double preprocess_times = 0.0;
    double infer_times      = 0.0;
    double decode_times     = 0.0;
    double endtoend_times   = 0.0;
    std::vector<double> time;
    int nums = preproccess_time.size();
    for ( int ind = 0; ind < nums; ind++ ) {
      preprocess_times += preproccess_time[ind];
      infer_times      += infer_time[ind];
      decode_times     += decode_time[ind];
      endtoend_times   += endtoend_time[ind];

    }

    auto average_pretime = preprocess_times / nums;
    auto average_inftime = infer_times / nums;
    auto average_dectime = decode_times / nums;
    auto average_endtoendtime = endtoend_times / nums;
    time.push_back(average_pretime);
    time.push_back(average_inftime);
    time.push_back(average_dectime);
    time.push_back(average_endtoendtime);
    time.push_back(endtoend_times);
  return time;
}

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_COMMON_H__
