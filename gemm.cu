#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// RG*RG*MAXN must fit within mytype

#define MAXN 100000
#define RG 10
#define USECPSEC 1000000ULL
#define nTPB 256

static inline int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

typedef double mytype;

unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

__global__ void gemm_Kernel2(const mytype * A, const mytype * B, mytype *out, const int N){
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = blockIdx.y;
    if (x <= N && y <= N){
      mytype my_sum = 0;
      for (int i = 0; i < N; i++) {
          my_sum += __ldg(A + i + y * N) * __ldg(B + x + i * N);
      }
      out[x + y * N] = my_sum;
    }
}

__global__ void gemm_Kernel1(const mytype * A, const mytype * B, mytype *out, const int N){
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = blockIdx.y;
    if (x <= N && y <= N){
      mytype my_sum = 0;
      for (int i = 0; i < N; i++) {
          my_sum += A[i + y * N] * B[x + i * N];
        }
      out[x + y * N] = my_sum;
    }
}


int main(int argc, char *argv[]){
  mytype *d1_A, *d1_result, *result, *d1_B, *A, *B;
  mytype *d2_A, *d2_result, *d2_B;
  if (argc != 2) {printf("must specify N on the command line\n"); return 1;}
  int my_N = atoi(argv[1]);
  if ((my_N < 1) || (my_N > MAXN)) {printf("N out of range\n"); return 1;}
  B   = (mytype *)malloc(my_N*my_N*sizeof(mytype));
  A   = (mytype *)malloc(my_N*my_N*sizeof(mytype));
  result   = (mytype *)malloc(my_N*my_N*sizeof(mytype));

  cudaMalloc(&d1_B, my_N*my_N*sizeof(mytype));
  cudaMalloc(&d1_A, my_N*my_N*sizeof(mytype));
  cudaMalloc(&d1_result, my_N*my_N*sizeof(mytype));

  cudaMalloc(&d2_B, my_N*my_N*sizeof(mytype));
  cudaMalloc(&d2_A, my_N*my_N*sizeof(mytype));
  cudaMalloc(&d2_result, my_N*my_N*sizeof(mytype));

  for (int i=0; i < my_N*my_N; i++){
    A[i] = rand()%RG;
    B[i] = rand()%RG;
  }

  for (int i=0; i < my_N*my_N; i++){
    result[i]   = 0;
  }

  cudaMemset(d1_result, 0, my_N*my_N*sizeof(mytype));
  cudaMemset(d2_result, 0, my_N*my_N*sizeof(mytype));

  int loop = 100;
  unsigned long long k1_time = 0;
  dim3 grid(div_up(my_N, nTPB), my_N/1);
  for (int i = 0; i < loop; i++) {
    unsigned long long gpu_time = dtime_usec(0);
    cudaMemcpy(d1_A, A, my_N*my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_B, B, my_N*my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    gemm_Kernel1<<<grid,nTPB>>>(d1_A, d1_B, d1_result, my_N);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d1_result, my_N*my_N*sizeof(mytype), cudaMemcpyDeviceToHost);
    gpu_time = dtime_usec(gpu_time);
    k1_time += gpu_time;
  }

  unsigned long long k2_time = 0;
  for (int i = 0; i < loop; i++) {
    unsigned long long gpu_time = dtime_usec(0);
    cudaMemcpy(d2_A, A, my_N * my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(d2_B, B, my_N * my_N* sizeof(mytype), cudaMemcpyHostToDevice);
    gemm_Kernel2<<<grid,nTPB>>>(d2_A, d2_B, d2_result, my_N);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d2_result, my_N*my_N*sizeof(mytype), cudaMemcpyDeviceToHost);
    gpu_time = dtime_usec(gpu_time);
    k2_time += gpu_time;
  }

  printf("Finished. gemm1(without ldg) time: %ldus, gemm2(with ldg) time: %ldus\n", k1_time/100, k2_time/100);
  return 0;
}
