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

## Code

|Function|Code|Input|Output|
|-|-|-|-|
|Train a CNN AlexNet model|keras_tf_train_mnist.py|dataset: MNIST|fp32_frozen.pb|
|Quantize the frozen PB model file|inc_quantize_model.py|dataset: MNIST<br>model: fp32_frozen.pb<br>yaml: alexnet.yaml|alexnet_int8_model.pb|
|Test performance|profiling_inc.py|model: fp32_frozen.pb|JSON file|
|Compare the performance|compare_perf.py|JSON file|Log file<br>PNG file|

**run_inc_ft_mnist_sample.sh** will call above python scripts to finish the demo.

## Getting Started with Intel® DevCloud

This article assumes you are familiar with Intel® DevCloud environment. To learn more about working with Intel® DevCloud, please refer to [Intel® DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html).
Specifically, this article assumes:

1. You have an Intel® DevCloud account.
2. You are familiar with usage of Intel® DevCloud, like login by SSH client..
3. Developers are familiar with Python, AI model training and inference based on Tensorflow*.

## Hardware Environment

This demo could be executed on any Intel CPU. But it's recommended to use 2nd Generation Intel® Xeon® Scalable Processors or newer, which include:

1. AVX512 intruction to speed up training AI model.

2. Intel® Deep Learning boost: Vector Neural Network Instruction (VNNI) to accelerate AI/DL Inference.

## Setup Running Environment

### Intel® DevCloud

We prepared the running enviroment by Conda in Intel® DevCloud for oneAPI.
Activate it by:

```
conda activate /data/oneapi_workshop/INC
```

If you want to setup your own Conda environment, please refer to next chapter: Customer Server.

### Customer Server

Set up own Conda enviroment in local server, cloud (including Intel® DevCloud):

#### Base on Intel® oneAPI AI Analytics Toolkit

1. Install Intel® oneAPI AI Analytics Toolkit.

For installation instructions, refer to [Intel® oneAPI Toolkits Installation Guides](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html).

2. Set up your Intel® oneAPI AI Analytics Toolkit environment.

Change the oneAPI installed path in the following command, according to your installation.

In this case, we use /opt/intel/oneapi.

```
source /opt/intel/oneapi/setvars.sh
```

3. Create Conda Envrionment: **user_tensorflow**

```
./set_env.sh
```

#### Install from Scrach by Conda

Create Conda Envrionment: **user_tensorflow** in Intel channel.

```
conda create -n user_tensorflow -c intel python=3.9 -y
conda activate user_tensorflow
conda install -n user_tensorflow -c intel tensorflow python-flatbuffers -y
conda install -n user_tensorflow -c intel neural-compressor -y
conda install -n user_tensorflow runipy notebook -y
```

## Update Script

Edit **run_inc_ft_mnist_sample.sh** to set the Conda enviroment according to above result: **/data/oneapi_workshop/INC** or **user_tensorflow**.

```
vi run_inc_ft_mnist_sample.sh

...
conda activate user_tensorflow
or
conda activate /data/oneapi_workshop/INC
...
```

## Run in Intel® DevCloud for oneAPI

### Run in Jupyter Notebook in Intel® DevCloud for oneAPI

Please open **inc_sample_for_tensorflow.ipynb** in Jupyter Notebook.

Following the guide to run this demo.

### Run in SSH Login Intel® DevCloud for oneAPI

This demo will show the obviously acceleration by VNNI. In Intel® DevCloud, please choose compute node with the property 'clx' or 'icx' or 'spr' which support VNNI.

#### Job Submit
```
!qsub run_inc_ft_mnist_sample.sh -d `pwd` -l nodes=1:icx:ppn=2
28029.v-qsvr-nda.aidevcloud
```

Note, please run above command in login node. There will be error as below if run it on compute node:
```
qsub: submit error (Bad UID for job execution MSG=ruserok failed validating uXXXXX/uXXXXX from s001-n054.aidevcloud)
```

#### Check job status

```
qstat
```

After the job is over (successfully or fault), there will be log files: 

1. **run_inc_ft_mnist_sample.sh.o28029**
2. **run_inc_ft_mnist_sample.sh.e28029**

### Check Result

#### Check Result in Log File

Check the result in log file: **run_inc_ft_mnist_sample.sh.o28029**:

```
!tail -23 run_inc_ft_mnist_sample.sh.o1842253


Compare the Performance of FP32 and INT8 Models
Model            FP32                     INT8                    
throughput(fps)  572.4982883964987        3218.52236638019        
latency(ms)      2.8339174329018104       1.9863116497896156      
accuracy(%)      0.9799                   0.9796                  

Save to fp32_int8_aboslute.png

Model            FP32                     INT8                    
throughput_times 1                        5.621889936815179       
latency_times    1                        0.7009066766478504      
accuracy_diff(%) 0                        -0.029999999999986926   

Save to fp32_int8_times.png
Please check the PNG files to see the performance!
This demo is finished successfully!
Thank you!

########################################################################
# End of output for job 1842253.v-qsvr-1.aidevcloud
# Date: Thu 27 Jan 2022 07:05:52 PM PST
########################################################################

...

```


We will see the performance and accuracy of FP32 and INT8 model. The performance could be obviously increased if running on Xeon with VNNI.

#### Check Result in PNG file

The demo creates figure files: fp32_int8_aboslute.png, fp32_int8_times.png to show performance bar. They could be used in report.

Copy files from DevCloud in host:

```
scp devcloud:~/intel_neural_compressor_sample/*.png ./
```

## License

Code samples are licensed under the MIT license. See
[License.txt](License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](third-party-programs.txt)
