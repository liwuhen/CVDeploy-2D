<!--
Copyright (c) Model-Infer Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

(l-main-doc-page)=

<div align="center">

<img src="./_static/cv-deploy-light-color.png" width="500" height="140">

</div>

<p style="text-align:center">
   <strong>AI model deployment based on NVIDIA and Qualcomm platforms
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/liwuhen/CVDeploy-2D" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/liwuhen/CVDeploy-2D/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/liwuhen/CVDeploy-2D/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>



The repository mainly provides 2D model inference functionality, and the code provides daily development of packaged libs for integration, testing, and inference. The framework provides multi-threaded, singleton pattern, producer and consumer patterns. Cache log analysis is also supported.


CVDeploy-2D Features:
* Supports real-time caching of runtime log files
* Offers easy integration via a singleton design pattern, providing header files and libraries to encapsulate the algorithm program
* Supports YAML configuration files to set essential parameters for model files
* Compatible with various hardware platforms, including NVIDIA and Qualcomm
* Supports multiple 2D models, such as YOLOv5 and YOLOX
* Integrates multithreading with producer-consumer mode to enable concurrent processing
* Includes user-friendly Bash scripts for one-click installation and execution

Planned Features for CVDeploy-2D:
* Support for model compression techniques, such as quantization.
* Memory leak detection and precision validation for ONNX, TRT, and QNN models.
* Integration of dynamic object detection with geometric tracking algorithms.


Documentation
-------------

```{toctree}
:maxdepth: 1
:caption: Getting Started

getting_started/installation
```

```{toctree}
:maxdepth: 1
:caption: Framework

intro/framework
```

```{toctree}
:maxdepth: 1
:caption: Algorithm

algorithm/Yolov5
algorithm/Yolox
```

```{toctree}
:maxdepth: 1
:caption: Hardware Platform

platform/nvidia
platform/qnn
```
