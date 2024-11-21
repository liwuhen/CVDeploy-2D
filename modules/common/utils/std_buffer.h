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

#ifndef APP_COMMON_STD_BUFFER_H__
#define APP_COMMON_STD_BUFFER_H__

#include <condition_variable>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace hpc {
namespace common {

template <typename T>
class MsgQueue {
 public:
  explicit MsgQueue(int length) { length_ = std::max(1, length); }
  ~MsgQueue() = default;
  void Push(const T& message);
  int Size() const;
  bool Empty() const;
  T WaitPop();
  bool Pop(T& message);
  void Clear();

 private:
  MsgQueue() = delete;
  MsgQueue(const MsgQueue& other) = delete;
  MsgQueue& operator=(const MsgQueue&);
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  int length_;
};

template <typename T>
void MsgQueue<T>::Push(const T& message) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push(message);
  while (static_cast<int>(queue_.size()) > length_) {
    queue_.pop();
  }
  lock.unlock();
  condition_.notify_one();
}

template <typename T>
bool MsgQueue<T>::Empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.empty();
}

template <typename T>
int MsgQueue<T>::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

template <typename T>
T MsgQueue<T>::WaitPop() {
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [this]() { return !queue_.empty(); });
  T message = queue_.front();
  queue_.pop();
  return message;
}

template <typename T>
bool MsgQueue<T>::Pop(T& message) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (queue_.empty()) return false;
  message = queue_.front();
  queue_.pop();
  return true;
}

template <typename T>
void MsgQueue<T>::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  while (!queue_.empty()) {
    queue_.pop();
  }
}

}  // namespace common
}  // namespace hpc

#endif  // APP_COMMON_STD_BUFFER_H__
