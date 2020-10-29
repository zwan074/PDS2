/******************************************************************************
Start up demo program for 159735 Assignment 3 Semester 1 2013
All this does is initialize the image and write it to a file.
To compile:
make heat_demo
To run (for example to make a 100X100 pixel image):
./heat_demo 100
******************************************************************************/
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>
#include "arrayff.hxx"
#include "draw.hxx"


int main(int argc, char* argv[]) 
{
  int NUM_THREADS =atoi(argv[2]) ;
  if (NUM_THREADS < 1) return 0;

  double T0, T1 ,T2;
  T0 = omp_get_wtime();
  T1 = omp_get_wtime();

  // X and Y dimensions. Force it to be a square.
  const int npix = atoi(argv[1]);
  const int npixx = npix;
  const int npixy = npix;
  const int ntotal = npixx * npixy;
  const float tol = 0.00001;
  // Images as 2D arrays: h is the current image, g is the updated
  // image. To access individual pixel elements, use the () operator. 
  // Note: that y is the first index (to reflect row major
  // order). Eg: h(y, x) = fubar
  Array<float, 2> h(npixy, npixx), g(npixy, npixx);

  const int nrequired = npixx * npixy;
  const int ITMAX = 1000000;

  int iter = 0;
  int shared_nconverged = 0 ;
  int nconverged = 0;  
  // Draw the printed circuit components
  fix_boundaries2<float>(h);
  
  omp_set_num_threads(NUM_THREADS);

  int  id,nthrds, p_start1,p_end1,p_start2,p_end2 , step;

  std::cout << "Seq Time : " << omp_get_wtime()-T0 << " time" << std::endl;
  
  T2 = omp_get_wtime();

  #pragma omp parallel private (T2, id,nthrds, p_start1,p_end1,p_start2,p_end2 , step,nconverged) 
{
  

  nthrds = omp_get_num_threads();
  step =  npixx / nthrds; 
  id = omp_get_thread_num();
  
  //one thread case
  if (nthrds == 1) {
    p_start1 = 1 ;
    p_start2 = 0 ;
    p_end1 = npixy - 1;
    p_end2 = npixy;
  }
  else if (id == 0) { 
    p_start1 = 1 ;
    p_start2 = 0 ;
    p_end1 = step;
    p_end2 = p_end1;
  }
  else if (id == (nthrds - 1) ) {
    p_start1 = id * step ;
    p_start2 = p_start1 ;
    p_end1 = npixy - 1;
    p_end2 = npixy;
  }
  else {
    p_start1 = id * step ;
    p_start2 = p_start1 ;
    p_end1 = (id + 1) * step;
    p_end2 = p_end1;
  }
  
  //std::cout << id << " " << p_start1 << " p_start1 " << p_end1 << " p_end1 " << std::endl;
  //std::cout << id << " " << p_start2 << " p_start2 " << p_end2 << " p_end2 " << std::endl;

  do {

    for (int y = p_start1; y < p_end1; ++y) {
      for (int x = 1; x < npixx-1; ++x) {
        g(y, x) = 0.25 * (h(y, x-1) + h(y, x+1) + h(y-1, x) + h(y+1,x));
      }
    }
    
    #pragma omp barrier
    
    #pragma omp single 
    {
      fix_boundaries2(g); 
      shared_nconverged = 0;
    } 

    nconverged = 0;

    for (int y = p_start2; y < p_end2; ++y) {
      for (int x = 0; x < npixx; ++x) {
        float dhg = std::fabs(g(y, x) - h(y, x));
        if (dhg < tol) {
            ++nconverged; 
        }
        h(y, x) = g(y, x);
      }
    }

    #pragma omp atomic 
      shared_nconverged += nconverged;
    
    #pragma omp single 
    {
      ++iter;
    }
    #pragma omp barrier
  } while (shared_nconverged < nrequired && iter < ITMAX);

  
}

  // This is the initial value image where the boundaries and printed
  // circuit components have been fixed
  dump_array<float, 2>(h, "plate0.fit");
  std::cout << "Parallel Time : " << omp_get_wtime()-T2 << " time" << std::endl;
  std::cout << "Required " << iter << " iterations" << std::endl;
  std::cout << "Required " << NUM_THREADS << " Threads" << std::endl;
  std::cout << "Total Time " << omp_get_wtime()-T1 << " time" << std::endl;
  // Complete the sequential version to compute the heat transfer,
  // then make a parallel version of it
}
