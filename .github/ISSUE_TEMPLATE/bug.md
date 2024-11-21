---
name: Bug report
about: Reporta bug to help improve this repository.
title: ''
labels: 'bug'
assignees: ''

---

# Bug Report

### Is the issue related to model conversion?


### Describe the bug


### System information
- OS Platform and Distribution (*e.g. Linux Ubuntu 20.04*):
- Software version (*e.g. Model-Infer-v1.0*):
- C++ version:
- GCC/Compiler version (if compiling from source):
- CMake version:
- Visual Studio version (if applicable):


### Reproduction instructions
Steps to reproduce a bug:

1. **Offer single test code**
```c++
int main(int argc, char* argv[]) {
    std::string path = argv[0];
    std::size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        path =  path.substr(0, pos);
    }

    std::string file_path_ = path + "/config/yaml/" + "yolox_config.yaml";
    std::shared_ptr<InterfaceYolo> inference = InterfaceYolo::getInstance();
    inference->InitConfig(path, file_path_);
    inference->Init();
    inference->Start();

    auto filesPath = path + "/config/data/coco/val2017";
    TestDemo(inference.get(), filesPath);

    inference->Stop();
    return 0;
}
```

2. **Provide ideas for reproducing the problem**

### Expected behavior
A clear and concise description of what you expected to happen.

### Additional Notes
Add your thoughts about the problem and the code (if you wish)
