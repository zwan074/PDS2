/* CUDA timing example

   To compile: nvcc -o testprog2 testprog2.cu

 */
#include <iostream>

#include <cuda.h>

// Kernel that executes on the CUDA device. This is executed by ONE
// stream processor
__global__ void vec_add(float* A, float* B, float* C, int N)
{
  // What element of the array does this thread work on
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) 
    C[i] = A[i] + B[i];
}

// main routine that executes on the host
int main(void)
{
  int n;
  int N = 10000000;
  size_t size = N * sizeof(float);

  // CUDA event types used for timing execution
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate in HOST memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  // Initialize vectors
  for (n = 0; n < N; ++n) {
    h_A[n] = 1 * n;
    h_B[n] = 1 * n;
  }

  // Allocate in DEVICE memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy vectors from host to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Set up layout of kernel grid
  int threadsPerBlock = 1024;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "Launching a grid of " 
	    << blocksPerGrid << " "
	    << threadsPerBlock * blocksPerGrid
	    << " threads" << std::endl;

  // Launch kernel and time it
  cudaEventRecord(start, 0);
  vec_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;  // Must be a float
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "Kernel took: " << time << " ms" << std::endl;

  // Copy result from device memory into host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Print the first and last 10 elements of the arrays
  for (n = 0; n < N; ++n) {
    if (n < 10 || n >= N - 10) 
      std::cout << n << " " << h_A[n] << " " << h_B[n] 
		<< " " << h_C[n] << std::endl;
  } 

  free(h_A);
  free(h_B);
  free(h_C);
}
