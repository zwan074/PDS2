/* 159.735 Semester 2, 2016.  Ian Bond, 3/10/2016
 
 Sequential version of the N-sphere counting problem for Assignment
 5. Two alternative algorithms are presented.

 Note: a rethink will be needed when implementing a GPU version of
 this. You can't just cut and paste code.

 To compile: g++ -O3 -o nsphere nsphere.cpp
 (you will get slightly better performance with the O3 optimization flag)
*/
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>

#include <vector>
#include <cuda.h>

using namespace std;

__global__ void inc(unsigned long long int *foo) {
  atomicAdd(foo, 1);
}

int main() {
  unsigned long long int count = 0, *cuda_count;
  cudaMalloc((void**)&cuda_count, sizeof(unsigned long long int));
  cudaMemcpy(cuda_count, &count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
  cout << "count: " << count << '\n';
  inc <<< 100, 25 >>> (cuda_count);
  cudaMemcpy(&count, cuda_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaFree(cuda_count);
  cout << "count: " << count << '\n';
  return 0;
}
