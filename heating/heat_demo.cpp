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

#define NUM_THREADS 5

int sum_vector(std::vector<int> vector) {
  int ret = 0 ;
  for (int i = 0 ; i < NUM_THREADS; i++) 
    ret += vector[i];
  return ret;
}

int main(int argc, char* argv[]) 
{
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

  std::vector<int> iter(NUM_THREADS) ;
  std::vector<int> nconverged(NUM_THREADS) ;
    
  // Draw the printed circuit components
  fix_boundaries2<float>(h);
  
  omp_set_num_threads(NUM_THREADS);
  double T0, T1;
  T0 = omp_get_wtime();

  #pragma omp parallel 
{
  int i, id,nthrds, p_start1,p_end1,p_start2,p_end2 , step;
  
  id = omp_get_thread_num();
  nthrds = omp_get_num_threads();
  
  step =  npixx / nthrds; 
  //one thread case
  if (nthrds == 1) {
    p_start1 = 1 ;
    p_start2 = 0 ;
    p_end1 = npixx - 1;
    p_end2 = npixx;
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
    p_end1 = npixx - 1;
    p_end2 = npixx;
  }
  else {
    p_start1 = id * step ;
    p_start2 = p_start1 ;
    p_end1 = (id + 1) * step;
    p_end2 = p_end1;
  }
  //std::cout << id << " " << step << " step " << std::endl;
  std::cout << id << " " << p_start1 << " p_start1 " << p_end1 << " p_end1 " << std::endl;
  std::cout << id << " " << p_start2 << " p_start2 " << p_end2 << " p_end2 " << std::endl;
  do {
    for (int y = p_start1; y < p_end1; ++y) {
      for (int x = 1; x < npixx-1; ++x) {
        g(y, x) = 0.25 * (h(y, x-1) + h(y, x+1) + h(y-1, x) + h(y+1,x));
      }
    }
    
    #pragma omp barrier
    

    #pragma omp single 
      fix_boundaries2(g); // doing once ?
    
      
    nconverged[id] = 0;

    for (int y = p_start2; y < p_end2; ++y) {
      for (int x = 0; x < npixx; ++x) {
        float dhg = std::fabs(g(y, x) - h(y, x));
        if (dhg < tol) {
            ++nconverged[id]; //affected by other processess?
        }
        h(y, x) = g(y, x);
      }
    }
    
    ++iter[id];
    //std::cout << id << " " << sum_vector(iter) << " iterations " << sum_vector(nconverged)  << " nconverged" << std::endl;
    #pragma omp barrier
  } while (sum_vector(nconverged) < nrequired && sum_vector(iter) / NUM_THREADS < ITMAX);

}
  
  T1 = omp_get_wtime();

  // This is the initial value image where the boundaries and printed
  // circuit components have been fixed
  dump_array<float, 2>(h, "plate0.fit");
  std::cout << "Required " << sum_vector(iter)  / NUM_THREADS << " iterations" << std::endl;
  std::cout << "Required " << NUM_THREADS << " Threads" << std::endl;
  std::cout << "Required " << T1-T0 << " time" << std::endl;
  // Complete the sequential version to compute the heat transfer,
  // then make a parallel version of it
}
