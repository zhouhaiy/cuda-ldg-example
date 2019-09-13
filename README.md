# cuda-ldg-example
How To Build:

./make.sh

How To Run:

./product 8192

Test Result: (64K Bytes read only memory access, Computation = 16K ops)

Finished. k1(without ldg) time: 967us, k2(with ldg) time: 927us

Summary

For 64K Bytes read only memory access, __ldg instrinsics can bring average 3% ~ %4 performance gain in this simple dot product kernel. (Computation 16K ops similar as vec product) 
