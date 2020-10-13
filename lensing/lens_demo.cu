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
 
 
 __global__ void lensim_gpu(float* xlens, float* ylens, float* eps, int npixx, int npixy ,int nlenses , float* lensim)
 {
   
   int i = blockDim.x * blockIdx.x + threadIdx.x;
 
   const float rsrc = 0.1;      // radius
   const float ldc  = 0.5;      // limb darkening coefficient
   const float xsrc = 0.0;      // x and y centre on the map
   const float ysrc = 0.0;
   const float lens_scale = 0.005;
   const float WL  = 2.0;
   const float XL1 = -WL;
   const float YL1 = -WL;
 
   const float rsrc2 = rsrc * rsrc;
   float xl, yl, xs, ys, sep2, mu;
   float xd, yd;
 
   int iy = i / npixy;
   int ix = i % npixx;
 
   yl = YL1 + iy * lens_scale;
   xl = XL1 + ix * lens_scale;
   //shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);
 
   float dx, dy, dr;
   xs = xl;
   ys = yl;
   for (int p = 0; p < nlenses; ++p) {
     dx = xl - xlens[p];
     dy = yl - ylens[p];
     dr = dx * dx + dy * dy;
     xs -= eps[p] * dx / dr;
     ys -= eps[p] * dy / dr;
   }
 
 
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
 
   size_t size1 = npixx_npixy * sizeof(float);
   size_t size2 = nlenses * sizeof(float);
   float *d_lensim,*d_xlens,*d_ylens,*d_eps;
 
   cudaMalloc(&d_lensim, size1);
   cudaMalloc(&d_xlens, size2);
   cudaMalloc(&d_ylens, size2);
   cudaMalloc(&d_eps, size2);
 
 
   cudaMemcpy(d_xlens, xlens, size2, cudaMemcpyHostToDevice);
   cudaMemcpy(d_ylens, ylens, size2, cudaMemcpyHostToDevice);
   cudaMemcpy(d_eps, eps, size2, cudaMemcpyHostToDevice);
   cudaMemcpy(d_lensim, lensim.buffer, size1, cudaMemcpyHostToDevice);
 
 
 
   int threadsPerBlock = 256;
   int blocksPerGrid = (npixx_npixy) / threadsPerBlock;
 
   lensim_gpu<<<blocksPerGrid, threadsPerBlock>>>( d_xlens,  d_ylens,  d_eps,  npixx, npixy , nlenses , d_lensim);
 
   cudaMemcpy( lensim.buffer, d_lensim, size1, cudaMemcpyDeviceToHost);
 
   cudaFree(d_lensim);
   cudaFree(d_xlens);
   cudaFree(d_ylens);
   cudaFree(d_eps);
 
 
   // Write the lens image to a FITS formatted file. You can view this
   // image file using ds9
   dump_array<float, 2>(lensim, "lens.fit");
 
   delete[] xlens;
   delete[] ylens;
   delete[] eps;
 }
 
   
 