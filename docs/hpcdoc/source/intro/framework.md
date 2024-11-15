# Framework  
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white) 

## 1. Model Inference
The inference framework is divided into three modules: pre-processing, inference, and decoding, and the object tracking part is another sub-thread.

The flow of the framework is shown below

![Framework Flow](./images/infer.jpg)

Frame output can be connected to middleware ros1, ros2, or other middleware.
