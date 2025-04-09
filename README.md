<div align="center">

<img src="./docs/images/cv-deploy-light-color.png" width="500" height="140">

<h2 align="center">AI model deployment based on NVIDIA and Qualcomm platforms</h2>


[<span style="font-size:20px;">**Architecture**</span>](./docs/framework.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Documentation**</span>](https://liwuhen.cn/CVDeploy-2D)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Blog**</span>](https://www.zhihu.com/column/c_1839603173800697856)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Roadmap**</span>](./docs/roadmap.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Slack**</span>](https://app.slack.com/client/T07U5CEEXCP/C07UKUA9TCJ)


---

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white)
![Parallel Computing](https://img.shields.io/badge/Parallel-Computing-orange?style=for-the-badge)
![HPC](https://img.shields.io/badge/HPC-High%20Performance%20Computing-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMiAxN3YtMmgtM3YtM2gydi0yaDJ2LTJoLTR2N2gtN3YtN0g4djhoLTNWM0gzdjE4aDE4di00eiIvPjwvc3ZnPg==)
![Performance](https://img.shields.io/badge/Performance-Optimized-red?style=for-the-badge)
![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

The repository mainly provides 2D model inference functionality, and the code provides daily development of packaged libs for integration, testing, and inference. The framework provides multi-threaded, singleton pattern, producer and consumer patterns. Cache log analysis is also supported.
</div>

# ![third-party](https://img.shields.io/badge/third-party-blue) Third-party Libraries

|Libraries|Eigen|Gflags|Glog|Yaml-cpp|Cuda|Cudnn|Tensorrt|Opencv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Version|3.4|2.2.2|0.6.0|0.8.0|11.4|8.4|8.4|3.4.5|

# Getting Started
Visit our documentation to learn more.
- [Installation](./docs/hpcdoc/source/getting_started/installation.md)
- [Quickstart](./docs/hpcdoc/source/getting_started/Quickstart.md)
- [Supported Models](./docs/hpcdoc/source/algorithm/Supported_Models.md)
- [Supported Object Tracking](./docs/hpcdoc/source/algorithm/Supported_Object_Tracking.md)

# Performances
|Model|Platform|Resolution|FPS|Memory|Cpu|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Yolov5|NVIDIA RTX4060|640x640|-|-|-|
|Yolov5|NVIDIA orin|640x640|-|-|-|
|Yolox|NVIDIA RTX4060|416x416|-|-|-|
|Yolox|NVIDIA orin|416x416|-|-|-|

# ![Contribute](https://img.shields.io/badge/how%20to%20contribute-project-brightgreen) Contributing
Welcome users to participate in these projects. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributing guideline.We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in Working Groups, Working Groups have most of their discussions on [Slack](https://app.slack.com/client/T07U5CEEXCP/C07UKUA9TCJ) or QQ (938558640).


# References
- [Yolox: https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [Ultralytics: https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Blog**：[Setup Environment](https://zhuanlan.zhihu.com/p/818205320)
