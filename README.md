# cuda-ldg-example
How To Build:
./make.sh

How To Run:
./gemm 1024
./conv 8192

Hardware Env:
CPU: SkyLake 8180
GPU: Tesla  

Test Result:
gemm(without ldg) time: 13571us, gemm(with ldg) time: 13458us
conv1d(without ldg) time: 967us, conv1d(with ldg) time: 927us


Findings:
For shape (1024, 1024) matrix multiply, without __ldg intrinsics it will bring overall 0.7% ~ 0.8% performance penalty for this simple GEMM kernel.
For conv1d, without __ldg intrinsics it will bring average 3% ~ 4% performance penalty for conv1d kernel.
