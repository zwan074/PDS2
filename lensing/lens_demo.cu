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

typedef struct { 
  int width; 
  int height; 
  float* elements; 
} 2d_Array;

// Global variables! Not nice style, but we'll get away with it here.

// Boundaries in physical units on the lens plane
const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;
// Source star parameters. You can adjust these if you like - it is
// interesting to look at the different lens images that result
const float rsrc = 0.1;      // radius
const float ldc  = 0.5;      // limb darkening coefficient
const float xsrc = 0.0;      // x and y centre on the map
const float ysrc = 0.0;

// Pixel size in physical units of the lens image. You can try finer
// lens scale which will result in larger images (and take more
// time).
const float lens_scale = 0.005;

// Used to time code. OK for single threaded programs but not for
// multithreaded programs. See other demos for hints at timing CUDA
// code.
double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks = clock1 - clock2;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  return diffms; // Time difference in milliseconds
}

__global__ void lensim_gpu(float* xlens, float* ylens, float* eps, int nlenses , 2d_Array lensim)
{
  int iy = blockDim.x * blockIdx.x + threadIdx.x;
  int ix = blockDim.y * blockIdx.y + threadIdx.y;

  const float rsrc2 = rsrc * rsrc;
  float xl, yl, xs, ys, sep2, mu;
  float xd, yd;

  yl = YL1 + iy * lens_scale;
  xl = XL1 + ix * lens_scale;
  shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);
  xd = xs - xsrc;
  yd = ys - ysrc;
  sep2 = xd * xd + yd * yd;
  if (sep2 < rsrc2) {
    mu = sqrt(1 - sep2 / rsrc2);
    lensim[iy * lensim.width + ix ] = 1.0 - ldc * (1 - mu);
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

  // Size of the lens image
  const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
  const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
  std::cout << "# Building " << npixx << "X" << npixy << " lens image" << std::endl;

  // Put the lens image in this array
  2d_Array lensim_array;
  2d_Array d_lensim_array;
  d_lensim_array.width = npixy ;
  d_lensim_array.height = npixx ;
  lensim_array.width = npixy ;
  lensim_array.height = npixx ;
  size_t size = d_lensim_array.width * d_lensim_array.height * sizeof(float); 
  cudaMalloc(&lensim_array.elements, size); 

 
  int blockWidth = 16;
  int gridWidth =lensim_array.width /blockWidth;

  dim3 dimBlock(blockWidth, blockWidth);
  dim3 dimGrid(gridWidth, gridWidth);
  lenssim_gpu<<<dimGrid, dimBlock>>>(xlens, ylens, eps, nlenses, lensim_array);

  // Copy result from device memory into host memory
  cudaMemcpy(lensim_array.elements, d_lensim_array.elements, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(lensim_array);

  clock_t tstart = clock();

  // Draw the lensing image map here. For each pixel, shoot a ray back
  // to the source plane, then test whether or or not it hits the
  // source star
  /*
  const float rsrc2 = rsrc * rsrc;
  float xl, yl, xs, ys, sep2, mu;
  float xd, yd;
  int numuse = 0;
  for (int iy = 0; iy < npixy; ++iy) 
  for (int ix = 0; ix < npixx; ++ix) { 
    
    // YOU NEED TO COMPLETE THIS SECTION OF CODE

    // need position on lens in physical units

    // shoot a ray back to the source plane - make the appropriate
    // call to shoot() in lenses.h

    // does the ray hit the source star?
    yl = YL1 + iy * lens_scale;
    xl = XL1 + ix * lens_scale;
    shoot(xs, ys, xl, yl, xlens, ylens, eps, nlenses);
    xd = xs - xsrc;
    yd = ys - ysrc;
    sep2 = xd * xd + yd * yd;
    if (sep2 < rsrc2) {
      mu = sqrt(1 - sep2 / rsrc2);
      lensim(iy, ix) = 1.0 - ldc * (1 - mu);
    }
  }*/
  Array<float, 2> lensim(npixy, npixx);

  clock_t tend = clock();
  double tms = diffclock(tend, tstart);
  //std::cout << "# Time elapsed: " << tms << " ms " << numuse << std::endl;

  // Write the lens image to a FITS formatted file. You can view this
  // image file using ds9
  dump_array<float, 2>(lensim, "lens.fit");

  delete[] xlens;
  delete[] ylens;
  delete[] eps;
}

