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

#include <iostream>
#include <time.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <memory>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <functional>
#include "enum_msg.h"
#include "colorprintf.h"

/**
 * @namespace hpc::common
 * @brief hpc::common
 */
namespace hpc{
namespace common{

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
static bool save_file(const std::string& file, const void* data, size_t length) 
{

  FILE* f = fopen(file.c_str(), "wb");
  if(!f) { return false; }

  if( data && length > 0 ) {
    if( fwrite(data, 1, length, f) != length ) {
      fclose(f);
      return false;
    }
  }
  fclose(f);
  return true;
}

} //namespace common
} //namespace hpc

#endif // APP_COMMON_COMMON_H__