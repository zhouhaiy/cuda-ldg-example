# cuda-ldg-example
How To Build:

./make.sh

How To Run:

./gemm 1024

Hardware Env:

CPU: SkyLake 8180
GPU: Tesla  

Test Result: (1024 * 1024 matrix multiply)

k1(without ldg) time: 13571us, k2(with ldg) time: 13458us


Findings:
For shape (1024, 1024) matrix multiply, without __ldg intrinsics it will bring overall 0.7% ~ 0.8% performance penalty for this simple GEMM kernel.
