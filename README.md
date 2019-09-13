# cuda-ldg-example
How To Build:

./make.sh

How To Run:

./conv 8192

Test Result: (64K read only memory access under 8192)

Finished. k1(without ldg) time: 967us, k2(with ldg) time: 927us

Summary

For 64K read only memory access, __ldg instrinsics can bring average 3% ~ %4 performance gain in this simple conv kernel.  
