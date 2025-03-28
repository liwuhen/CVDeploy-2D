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

#ifndef APP_COMMON_FILE_H__
#define APP_COMMON_FILE_H__

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <dirent.h>
#include <cstring>

namespace FileSystem {

using namespace std;

class DirectoryHandle {
public:
    explicit DirectoryHandle(const std::string& path) : dir_(opendir(path.c_str())) {
        if (!dir_) {
            throw std::runtime_error("Failed to open directory: " + path);
        }
    }

    ~DirectoryHandle() {
        if (dir_) {
            closedir(dir_);
        }
    }

    DIR* get() const { return dir_; }

private:
    DIR* dir_;
};


static std::string file_name(const string& path, bool include_suffix) {

        if (path.empty()) return "";

        int p = path.rfind('/');

#ifdef U_OS_WINDOWS
        int e = path.rfind('\\');
        p = std::max(p, e);
#endif
        p += 1;

        //include suffix
        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

static bool save_file(const string& file, const void* data, size_t length){

    FILE* f = fopen(file.c_str(), "wb");
    if (!f) return false;

    if (data and length > 0){
        if (fwrite(data, 1, length, f) not_eq length){
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

static bool save_file(const string& file, const string& data){
    return save_file(file, data.data(), data.size());
}

static bool save_file(const string& file, const vector<uint8_t>& data){
    return save_file(file, data.data(), data.size());
}

} // namespace FileSystem

#endif  // APP_COMMON_FILE_H__
