
#!/bin/bash

export CUDA_HOME=/usr/local/cuda-9.0
${CUDA_HOME}/bin/nvcc -arch=sm_60 -O3 -o product product.cu
