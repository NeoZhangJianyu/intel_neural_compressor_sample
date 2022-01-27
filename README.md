# Intel® Neural Compressor Sample for TensorFlow*


## Background
Low-precision inference can speed up inference obviously, by converting the fp32 model to int8 or bf16 model. Intel provides Intel® Deep Learning Boost technology in the Second Generation Intel® Xeon® Scalable Processors and newer Xeon®, which supports to speed up int8 and bf16 model by hardware.

Intel® Neural Compressor helps the user to simplify the processing to convert the fp32 model to int8/bf16.

At the same time, Intel® Neural Compressor will tune the quanization method to reduce the accuracy loss, which is a big blocker for low-precision inference.

Intel® Neural Compressor is released in Intel® AI Analytics Toolkit and works with Intel® Optimization of TensorFlow*.

Please refer to the official website for detailed info and news: [https://github.com/intel/neural-compressor](https://github.com/intel/neural-compressor)

## Introduction
This is a demo to show an End-To-End pipeline to build up a CNN model to recognize handwriting number and speed up AI model by Intel® Neural Compressor.

1. Train a CNN AlexNet model by Keras and Intel Optimization for Tensorflow based on dataset MNIST.

2. Quantize the frozen PB model file by Intel® Neural Compressor to INT8 model.

3. Compare the performance of FP32 and INT8 model by same script.


We will learn the acceleration of AI inference by Intel AI technology:

1. Intel® Deep Learning Boost

2. Intel® Neural Compressor

3. Intel® Optimization for Tensorflow*

## Getting Started with Intel® DevCloud
This article assumes you are familiar with Intel&reg; DevCloud environment. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html).
Specifically, this article assumes:

1. You have an Intel® DevCloud account.
2. You are familiar with usage of Intel® DevCloud, like login by SSH client..
3. Developers are familiar with Python, AI model training and inference based on Tensorflow*.

## Running Environment

### Hardware

This demo could be executed on any Intel CPU. But it's recommended to use 2nd Generation Intel® Xeon® Scalable Processors or newer, which include:

1. AVX512 intruction to speed up training AI model.

2. Intel® Deep Learning boost: Vector Neural Network Instruction (VNNI) to accelerate AI/DL Inference.

### Software

Setup Conda running environment based on oneAPI:

```
source ${ONEAPI_ROOT}/setvars.sh --force
./setup_env.sh
```

Conda Environment 'user_tensorflow' will be created. Check by:

```
conda info -e
# conda environments:
#
base                  *  /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest
2022.0.1                 /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/2022.0.1
pytorch                  /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/pytorch
pytorch-1.8.0            /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/pytorch-1.8.0
tensorflow               /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/tensorflow
tensorflow-2.6.0         /glob/development-tools/versions/oneapi/2022.1.1/oneapi/intelpython/latest/envs/tensorflow-2.6.0
user_tensorflow          /home/uXXXXX/.conda/envs/user_tensorflow
```

Activate 'user_tensorflow':

```
conda activate user_tensorflow

(user_tensorflow) uXXXXX@s001-n054:~$ 

```



## Run in Intel® DevCloud

This demo will show the obviously acceleration by VNNI. In Intel® DevCloud, please choose compute node with the property 'clx' or 'icx' or 'spr' which support VNNI.

```
qsub -l nodes=1:icx:ppn=2 -d . run_inc_ft_mnist_sample.sh
28029.v-qsvr-nda.aidevcloud
```

After the job is over (successfully or fault), there will be log files: 

1. **run_inc_ft_mnist_sample.sh.o28029**
2. **run_inc_ft_mnist_sample.sh.e28029**

Check the running result in **run_inc_ft_mnist_sample.sh.o28029**:

```
cat run_inc_ft_mnist_sample.sh.o28029

...

Model            FP32                     INT8                    
throughput(fps)  XXX.2996991121573242     YYYY.2343232442242   
latency(ms)      X.555093278690261122     Y.8621371522241709      
accuracy(%)      0.9821                   Y.9822                  

Save to fp32_int8_aboslute.png

Model            FP32                     INT8                    
throughput_times 1                        Y.8235927757156256      
latency_times    1                        Y.4911037147667566      
accuracy_diff(%) 0                        Y.0100000000000051   

Save to fp32_int8_times.png

...

```

We will see the performance and accuracy of FP32 and INT8 model. The performance could be obviously increased if running on Xeon with VNNI.

The demo creates figure files: fp32_int8_aboslute.png, fp32_int8_times.png to show performance bar. They could be used in report.


Note, please run above command in login node. There will be error as below if run it on compute node:
```
qsub: submit error (Bad UID for job execution MSG=ruserok failed validating uXXXXX/uXXXXX from s001-n054.aidevcloud)
```


## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt)





