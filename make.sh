
#!/bin/bash

export CUDA_HOME=/usr/local/cuda-9.0
${CUDA_HOME}/bin/nvcc -arch=sm_60 -O3 -o gemm gemm.cu
${CUDA_HOME}/bin/nvcc -arch=sm_60 -O3 -o conv conv.cu
