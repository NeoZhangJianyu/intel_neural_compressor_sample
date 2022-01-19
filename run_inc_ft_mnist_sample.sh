#!/bin/bash

echo "Train Model by Keras/Tensorflow with MNIST"
python keras_tf_train_mnist.py
echo "Training is finished"

echo "Enable Intel Optimization for Tensorflow by exporting TF_ENABLE_MKL_NATIVE_FORMAT=0"
echo "Intel Optimized TensorFlow 2.5.0 and later require to set environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 before running Intel® Neural Compressor quantize Fp32 model or deploying the quantized model."

export TF_ENABLE_MKL_NATIVE_FORMAT=0

echo "Quantize Model by Intel Neural Compressor"
python inc_quantize_model.py
echo "Quantization is finished"

echo "Execute the profiling_inc.py with FP32 model file"
python profiling_inc.py --input-graph=./fp32_frezon.pb --omp-num-threads=4 --num-inter-threads=1 --num-intra-threads=4 --index=32
echo "FP32 performance test is finished"

echo "Execute the profiling_inc.py with INT8 model file"
python profiling_inc.py --input-graph=./alexnet_int8_model.pb --omp-num-threads=4 --num-inter-threads=1 --num-intra-threads=4 --index=8
echo "INT8 performance test is finished"

echo "Compare the Performance of FP32 and INT8 Models"
python compare_perf.py
echo "Please check the PNG files to see the performance!"

if [[ $? -eq 0 ]]
then
  echo "This demo is finished successfully!"
else
  echo "This demo is fault!"
fi

echo "Thank you!"