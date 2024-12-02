<!--
Copyright (c) CV-Deploy Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Framework

## 1. Model Inference
The inference framework is divided into three modules: pre-processing, inference, and decoding, and the object tracking part is another sub-thread.

The flow of the framework is shown below

![Framework Flow](./images/infer.jpg)

Module description:
- Pre-processing module: contains receiving external signals, receiving external images, image pre-processing, image encoding, image decoding, image format conversion, etc.;
- Reasoning module: It contains memory opening, model loading, model reasoning, model releasing, etc. for onnx exported TensorRT model module;
- Post-processing module: contains decoding the inference results, post-decoding processing, result output and visualization, etc.

Thread description:
- Pre-processing module, inference module and decoding module are placed in a consumer thread, and the producer is the buffer that receives external images, which is in the main thread;
- Some other visual logic post-processing, e.g., object tracking module, is placed in another consumer thread.


Frame output can be connected to middleware ros1, ros2, or other middleware.
