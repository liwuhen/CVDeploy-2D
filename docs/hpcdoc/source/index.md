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
   <a class="github-button" href="https://github.com/liwuhen/Model-Infer" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/liwuhen/Model-Infer/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/liwuhen/Model-Infer/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>



The repository mainly provides model inference functionality, and the code provides daily development of packaged libs for integration, testing, and inference. The framework provides multi-threaded, singleton pattern, producer and consumer patterns. Cache log analysis is also supported. At the same time, this repository also supports autopilot driving, parking, and cockpit areas.

vLLM is fast with:

* State-of-the-art serving throughput
* Efficient management of attention key and value memory with **PagedAttention**
* Continuous batching of incoming requests
* Fast model execution with CUDA/HIP graph
* Quantization: `GPTQ <https://arxiv.org/abs/2210.17323>`_, `AWQ <https://arxiv.org/abs/2306.00978>`_, INT4, INT8, and FP8
* Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
* Speculative decoding
* Chunked prefill

vLLM is flexible and easy to use with:

* Seamless integration with popular HuggingFace models
* High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
* Tensor parallelism and pipeline parallelism support for distributed inference
* Streaming outputs
* OpenAI-compatible API server
* Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs, Gaudi® accelerators and GPUs, PowerPC CPUs, TPU, and AWS Trainium and Inferentia Accelerators.
* Prefix caching support
* Multi-lora support

For more information, check out the following:

* `vLLM announcing blog post <https://vllm.ai>`_ (intro to PagedAttention)
* `vLLM paper <https://arxiv.org/abs/2309.06180>`_ (SOSP 2023)
* `How continuous batching enables 23x throughput in LLM inference while reducing p50 latency <https://www.anyscale.com/blog/continuous-batching-llm-inference>`_ by Cade Daniel et al.
* :ref:`vLLM Meetups <meetups>`.


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
