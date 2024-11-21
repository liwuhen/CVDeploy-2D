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

#include "trt_infer.h"

namespace hpc {
namespace appinfer {

TrtInfer::TrtInfer() {}

TrtInfer::~TrtInfer() {}

/**
 * @description: init．
 */
bool TrtInfer::Init() {
  // onnx model loading, trt model generation
  if (!isFileExists_stat(parsemsgs_->trt_path_)) {
    GLOG_ERROR("Trt model does not exist. ");
    BuildModel();
    return false;
  }

  // parse trt model
  if (!ParseModel()) {
    GLOG_ERROR("Parse trt model failed. ");
    return false;
  }

  // allocator memcory
  if (!MemAllocator()) {
    GLOG_ERROR("Memory allocator failed. ");
    return false;
  }

  GLOG_INFO("[Init]: Trt infer module init ");
  return true;
}

/**
 * @description: The inference algorithm handles threads．
 */
bool TrtInfer::RunStart() {
  GLOG_INFO("[RunStart]: Trt infer module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool TrtInfer::RunStop() {
  GLOG_INFO("[RunStop]: Trt infer module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool TrtInfer::RunRelease() {
  GLOG_INFO("[RunRelease]: TrtInfer module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool TrtInfer::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  }

  GLOG_INFO("[SetParam]: Trt infer module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool TrtInfer::DataResourceRelease() {}

/**
 * @description: Inference.
 */
bool TrtInfer::Inference(float* output_img_device) {
  checkRuntime(cudaMemcpy(gpu_buffers_[engine_name_size_[binding_names_["input"][0]].first],
                          output_img_device, parsemsgs_->dstimg_size_ * sizeof(uint8_t),
                          cudaMemcpyDeviceToDevice));

  bool success =
      execution_context_->enqueueV2(reinterpret_cast<void**>(gpu_buffers_), stream_, nullptr);
  if (!success) {
    LOG(ERROR) << " Inference failed ";
    return false;
  }

  checkRuntime(cudaMemcpyAsync(
      cpu_buffers_[0], gpu_buffers_[engine_name_size_[binding_names_["output"][0]].first],
      sizeof(float) * engine_name_size_[binding_names_["output"][0]].second, cudaMemcpyDeviceToHost,
      stream_));
  checkRuntime(cudaStreamSynchronize(stream_));

  return true;
}

/**
 * @description: Build trt model from onnx.
 */
bool TrtInfer::BuildModel() {
  GLOG_INFO("=====> Build TensorRT Engine <===== ");
  //  Define builder, config and network.
  auto builder = make_nvshared(createInferBuilder(gLogger));
  if (!builder) {
    GLOG_ERROR("Can not create builder. ");
    return false;
  }
  const auto explicitBatch =
      1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = make_nvshared(builder->createNetworkV2(explicitBatch));
  if (!network) {
    GLOG_ERROR("Can not create network. ");
    return false;
  }
  auto config = make_nvshared(builder->createBuilderConfig());
  if (!config) {
    GLOG_ERROR("Can not create config. ");
    return false;
  }

  auto parser = make_nvshared(nvonnxparser::createParser(*network, gLogger));
  if (!parser->parseFromFile(parsemsgs_->onnx_path_.c_str(), 1)) {
    GLOG_ERROR("Failed to parse " << parsemsgs_->onnx_path_.c_str());
    return false;
  }

  int max_workspace_size = 1 << 28;
  GLOG_ERROR("Set max workspace size = " << max_workspace_size / 1024.0f / 1024.0f << "MB");
  config->setMaxWorkspaceSize(1 << 28);

  switch ((ModelACC)parsemsgs_->model_acc_) {
    case ModelACC::MODEL_FLOAT32:
      if (builder->platformHasTf32()) {
        config->setFlag(BuilderFlag::kTF32);
      }
      break;
    case ModelACC::MODEL_FLOAT16:
      if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
      }
      break;
    case ModelACC::MODEL_INT8:
      if (builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
      }
      break;
    default:
      break;
  }

  GLOG_INFO("Build model acc[0-fp32, 1-fp16, 2-int8]:  " << parsemsgs_->model_acc_);
  int maxBatchSize = 10;
  auto profile = builder->createOptimizationProfile();
  auto input_tensor = network->getInput(0);
  auto input_dims = input_tensor->getDimensions();

  input_dims.d[0] = 1;
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
  input_dims.d[0] = maxBatchSize;
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
  config->addOptimizationProfile(profile);

  auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
  if (engine == nullptr) {
    GLOG_ERROR("Build engine failed. ");
    return false;
  }

  // 将模型序列化，并储存为文件
  auto seridata = make_nvshared(engine->serialize());
  save_file(parsemsgs_->trt_path_, seridata->data(), seridata->size());
  GLOG_INFO("Build engine success.  ");
  return true;
}

/**
 * @description: Parse model.
 */
bool TrtInfer::ParseModel() {
  GLOG_INFO("=====> Begin Deserialize Engine <===== ");
  checkRuntime(cudaStreamCreate(&stream_));
  auto engine_data = LoadFile(parsemsgs_->trt_path_);
  if (engine_data.empty()) {
    GLOG_ERROR("Build engine failed  " << parsemsgs_->trt_path_);
    return false;
  }

  auto runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger));
  auto engine = unique_ptr<ICudaEngine, NvInferDeleter>(
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
  if (engine == nullptr) {
    GLOG_ERROR("Deserialize cuda engine failed! ");
    runtime->destroy();
    return false;
  }

  execution_context_ =
      std::unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
  if (execution_context_ == nullptr) {
    GLOG_ERROR("Failed to create context! ");
    return false;
  }

  // 明确当前推理时，使用的数据输入大小
  auto input_dims = engine->getBindingDimensions(0);
  input_dims.d[0] = parsemsgs_->batchsizes_;
  execution_context_->setBindingDimensions(0, input_dims);

  int nb_bindings = engine->getNbBindings();
  int in_size = 0, out_size = 0;
  for (int i = 0; i < nb_bindings; i++) {
    size_t size(1);
    string name = execution_context_->getEngine().getBindingName(i);
    auto dim = execution_context_->getBindingDimensions(i);

    // < tensorrt 8.5
    switch (execution_context_->getEngine().bindingIsInput(i)) {
      case false:
        out_size++;
        binding_names_["output"].push_back(name);
        break;
      case true:
        in_size++;
        binding_names_["input"].push_back(name);
        break;
    }

    for (int j = 0; j < dim.nbDims; j++) {
      size *= dim.d[j];
    }

    in_out_size_["input"] = in_size;
    in_out_size_["output"] = out_size;

    engine_name_size_.emplace(name, make_pair(i, size));
  }

  GLOG_INFO("[Model]: " + std::string(MODEL_FLAG));

  return true;
}

/**
 * @description: Memory allocator.
 */
bool TrtInfer::MemAllocator() {
  GLOG_INFO("Begin allocator memory ");
  // Allocate model input memory
  for (int i = 0; i < in_out_size_["input"]; i++) {
    checkRuntime(cudaMalloc(&gpu_buffers_[i],
                            sizeof(float) * engine_name_size_[binding_names_["input"][i]].second));
  }

  // Allocating memory for output data.（host）
  for (int i = 0; i < in_out_size_["output"]; i++) {
    auto out_node_size = engine_name_size_[binding_names_["output"][i]].second;
    checkRuntime(cudaMallocHost(&cpu_buffers_[i], sizeof(float) * out_node_size));

    int out_index = in_out_size_["input"] + i;
    checkRuntime(cudaMalloc(&gpu_buffers_[out_index], sizeof(float) * out_node_size));
  }

  GLOG_INFO("Memory allocator done ");

  return true;
}

/**
 * @description: Cpu and gpu memory free.
 */
bool TrtInfer::MemFree() {
  // free memory
  checkRuntime(cudaStreamDestroy(stream_));
  stream_ = nullptr;

  for (int out_index = 0; out_index < in_out_size_["output"]; out_index++) {
    checkRuntime(cudaFreeHost(cpu_buffers_[out_index]));
    cpu_buffers_[out_index] = nullptr;
  }

  for (int index = 0; index < in_out_size_["input"] + in_out_size_["output"]; index++) {
    checkRuntime(cudaFree(gpu_buffers_[index]));
    gpu_buffers_[index] = nullptr;
  }

  return true;
}

/**
 * @description: Load file.
 */
std::vector<uint8_t> TrtInfer::LoadFile(const string& file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();
  vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read(reinterpret_cast<char*>(&data[0]), length);
  }
  in.close();
  return data;
}

}  // namespace appinfer
}  // namespace hpc
