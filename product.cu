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

typedef double mytype;

void product(const mytype *A, const mytype *B, mytype* out, int N) {

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            out[i + j] += A[i] * B[j];
}

unsigned long long dtime_usec(unsigned long long prev){
  timeval tv1;
  gettimeofday(&tv1,0);
  return ((tv1.tv_sec * USECPSEC)+tv1.tv_usec) - prev;
}

__global__ void product_Kernel2(const mytype * A, const mytype * B, mytype *out, const int N){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < (2*N)-1){
      mytype my_sum = 0;
      for (int i = 0; i < N; i++)
        if (((idx < N) && (i <= idx)) || ((idx >= N) && (i > (idx-N)))) {
          my_sum += __ldg(A + i)*__ldg(B + idx - i);
        }
      out[idx] = my_sum;
    }
}

__global__ void product_Kernel1(const mytype * A, const mytype * B, mytype *out, const int N){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < (2*N)-1){
      mytype my_sum = 0;
      for (int i = 0; i < N; i++)
        if (((idx < N) && (i <= idx)) || ((idx >= N) && (i > (idx-N)))) {
          my_sum += A[i] * B[idx-i];
        }
      out[idx] = my_sum;
    }
}


int main(int argc, char *argv[]){
  mytype *h_A, *d_A, *h_result, *d_result, *result, *h_B, *d_B, *A, *B;
  if (argc != 2) {printf("must specify N on the command line\n"); return 1;}
  int my_N = atoi(argv[1]);
  if ((my_N < 1) || (my_N > MAXN)) {printf("N out of range\n"); return 1;}
  B   = (mytype *)malloc(my_N*sizeof(mytype));
  A   = (mytype *)malloc(my_N*sizeof(mytype));
  h_A = (mytype *)malloc(my_N*sizeof(mytype));
  h_B = (mytype *)malloc(my_N*sizeof(mytype));
  h_result = (mytype *)malloc(2*my_N*sizeof(mytype));
  result   = (mytype *)malloc(2*my_N*sizeof(mytype));

  cudaMalloc(&d_B, my_N*sizeof(mytype));
  cudaMalloc(&d_A, my_N*sizeof(mytype));
  cudaMalloc(&d_result, 2*my_N*sizeof(mytype));

  for (int i=0; i < my_N; i++){
    A[i] = rand()%RG;
    B[i] = rand()%RG;
    h_A[i] = A[i];
    h_B[i] = B[i];}

  for (int i=0; i < 2*my_N; i++){
    result[i]   = 0;
    h_result[i] = 0;}

  unsigned long long cpu_time = dtime_usec(0);
  product(A, B, result, my_N);
  cpu_time = dtime_usec(cpu_time);

  cudaMemset(d_result, 0, 2*my_N*sizeof(mytype));

  int loop = 100;
  unsigned long long k1_time = 0;
  for (int i = 0; i < loop; i++) {
    unsigned long long gpu_time = dtime_usec(0);
    cudaMemcpy(d_A, h_A, my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    product_Kernel1<<<((2*(my_N-1))+nTPB-1)/nTPB,nTPB>>>(d_A, d_B, d_result, my_N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, 2*my_N*sizeof(mytype), cudaMemcpyDeviceToHost);
    gpu_time = dtime_usec(gpu_time);
    k1_time += gpu_time;
  }

  unsigned long long k2_time = 0;
  for (int i = 0; i < loop; i++) {
    unsigned long long gpu_time = dtime_usec(0);
    cudaMemcpy(d_A, h_A, my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, my_N*sizeof(mytype), cudaMemcpyHostToDevice);
    product_Kernel2<<<((2*(my_N-1))+nTPB-1)/nTPB,nTPB>>>(d_A, d_B, d_result, my_N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, 2*my_N*sizeof(mytype), cudaMemcpyDeviceToHost);
    gpu_time = dtime_usec(gpu_time);
    k2_time += gpu_time;
  }
#if 0
  for (int i = 0; i < 2*my_N; i++) 
    if (result[i] != h_result[i]) 
    {
      printf("mismatch2 at %d, cpu: %d, gpu %d\n", i, result[i], h_result[i]); 
      return 1;
    }
#endif
  printf("Finished. k1(without ldg) time: %ldus, k2(with ldg) time: %ldus\n", k1_time/100, k2_time/100);
  return 0;
}
