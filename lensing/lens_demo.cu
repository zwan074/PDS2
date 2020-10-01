/* 
   159735 Parallel Programming

   Startup program for sequential implementation of simulation by ray
   tracing of gravitational lensing.
 */
#include <ctime>

#include <iostream>
#include <string>

#include <cmath>

#include "lenses.h"
#include "arrayff.hxx"
#include <cuda.h>

// Global variables! Not nice style, but we'll get away with it here.

// Boundaries in physical units on the lens plane
const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

__global__ void lensim_gpu(float* xlens, float* ylens, float* eps, int npixx_npixy ,int nlenses , float* lensim)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  const float rsrc = 0.1;      // radius
  const float ldc  = 0.5;      // limb darkening coefficient
  const float xsrc = 0.0;      // x and y centre on the map
  const float ysrc = 0.0;
  const float lens_scale = 0.005;

  const float rsrc2 = rsrc * rsrc;
  float xl, yl, xs, ys, sep2, mu;
  float xd, yd;

  int iy = npixx_npixy / i; 
  int ix = npixx_npixy % i;

  yl = YL1 + iy * lens_scale;
  xl = XL1 + ix * lens_scale;
  shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);
  xd = xs - xsrc;
  yd = ys - ysrc;
  sep2 = xd * xd + yd * yd;
  if (sep2 < rsrc2) {
    mu = sqrt(1 - sep2 / rsrc2);
    lensim[i] = 1.0 - ldc * (1 - mu);
  }

}

int main(int argc, char* argv[]) 
{
  // Set up lensing system configuration - call example_1, _2, _3 or
  // _n as you wish. The positions and mass fractions of the lenses
  // are stored in these arrays
  float* xlens;
  float* ylens;
  float* eps;
  const int nlenses = set_example_1(&xlens, &ylens, &eps);
  std::cout << "# Simulating " << nlenses << " lens system" << std::endl;
  const float lens_scale = 0.005;

  // Size of the lens image
  const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
  const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
  std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;
  const int npixx_npixy = npixx * npixy;

  // Put the lens image in this array
  Array<float, 2> lensim(npixy, npixx);


  size_t size = npixx_npixy * sizeof(float);
  float *d_lensim;
  
  cudaMalloc(&d_lensim, size);
  int threadsPerBlock = 1024;
  int blocksPerGrid = (npixx_npixy + threadsPerBlock - 1) / threadsPerBlock;

  lensim_gpu<<<blocksPerGrid, threadsPerBlock>>>( xlens,  ylens,  eps,  npixx_npixy , nlenses , d_lensim);
  cudaMemcpy( &lensim(0, 0), d_lensim, size, cudaMemcpyDeviceToHost);

  cudaFree(d_lensim);

  // Write the lens image to a FITS formatted file. You can view this
  // image file using ds9
  dump_array<float, 2>(lensim, "lens.fit");

  delete[] xlens;
  delete[] ylens;
  delete[] eps;
}

