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

#ifndef APP_YOLO_WARPAFFINE_H__
#define APP_YOLO_WARPAFFINE_H__

#include <cuda_runtime.h>

#include <iostream>

#include "Eigen/Dense"
#include "enum_msg.h"
#include "task_struct.hpp"

#define GPU_BLOCK_THREADS 512

using namespace Eigen;
using namespace hpc::common;

struct Norm {
  float mean[3] = {123.675, 116.28, 103.53};
  float std[3] = {58.395, 57.12, 57.375};
};

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs * 10 + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

void warp_affine_bilinear(uint8_t* src, int batch, InfertMsg& input_msg, float* dst, int dst_width,
                          int dst_height, uint8_t const_value, cudaStream_t stream, AppYolo mode);

#endif  // APP_YOLO_WARPAFFINE_H__
