# cuda-ldg-example
How To Build:

./make.sh

How To Run:

./product 8192

Test Result: (64K Bytes read only memory access, Computation = 64M ops)

Finished. k1(without ldg) time: 967us, k2(with ldg) time: 927us

Summary

For 64K Bytes read only memory access and 64M op computation, __ldg instrinsics can bring average 3% ~ %4 performance gain in this simple dot product kernel. (Memory access O(N), Computation O(square of N)) 
