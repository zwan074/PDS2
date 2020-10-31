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
#include <ctime>

double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks = clock1 - clock2;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  return diffms; // Time difference in milliseconds
}

long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
  long p = 1;
  for (long i = 0; i < k; ++i) p *= n;
  return p;
}

/*----------------------------------------------------------------------------*/


__global__ void count_in_v1_gpu (long ntotal , long base, long halfb, double rsquare, long ndim , unsigned long long int* count )
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;

  if (n >= ntotal)
    return;
  // Indices in x,y,z,.... 
  
  //long* index = (long*)malloc(ndim * sizeof(long));

  //for (long i = 0; i < ndim; ++i) index[i] = 0;
  
  long idx = 0;

  double rtestsq = 0;
  while (n != 0) {
    long rem = n % base;
    n = n / base;
    double xk = rem - halfb;
    rtestsq += xk * xk;
    //index[idx] = rem;
    ++idx;
  }


  for (long k = idx; k < ndim; ++k) {
    double xk = 0.0 - halfb;
    rtestsq += xk * xk;
  }

  if (rtestsq < rsquare) 
    atomicAdd(count,1);

}



void convert(long num, long base, std::vector<long>& index)
/* Convert a decimal number into another base system - the individual
   digits in the new base are stored in the index array. */
{
  const long ndim = index.size();
  for (long i = 0; i < ndim; ++i) index[i] = 0;
  long idx = 0;
  while (num != 0) {
    long rem = num % base;
    num = num / base;
    index[idx] = rem;
    ++idx;
  }
}

long count_in_v1(long ndim, double radius)
/* 
   Version 1 of the counting algorithm. Given:

   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere

   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/
{
  const long halfb = static_cast<long>(floor(radius));
  const long base = 2 * halfb + 1;
  const double rsquare = radius * radius;

  // This is the total number of points we will need to test.
  const long ntotal = powlong(base, ndim);

  long count = 0;

  // Indices in x,y,z,.... 
  std::vector<long> index(ndim, 0);

  // Loop over the total number of points. For each visit of the loop,
  // we covert n to its equivalent in a number system of given "base".
  for (long n = 0; n < ntotal; ++n) {
    convert(n, base, index);
    double rtestsq = 0;
    for (long k = 0; k < ndim; ++k) {
      double xk = index[k] - halfb;
      rtestsq += xk * xk;
    }
    if (rtestsq < rsquare) ++count;
  }

  return count;
}

/*----------------------------------------------------------------------------*/

void addone(std::vector<long>& index, long base, long i)
/* Add one to a digital counter of given base. When one digit hits
   maximum, it is necessary to carry one over into the next
   column. This is done recursively here. */
{
  long ndim = index.size();
  long newv = index[i] + 1;
  if (newv >= base) {
    index[i] = 0;
    if (i < ndim - 1) addone(index, base, i+1);
  }
  else {
    index[i] = newv;
  }
}

long count_in_v2(long ndim, double radius)
/* 
   Version 2 of the counting algorithm. Given:

   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere

   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/
{
  const long halfb = static_cast<long>(floor(radius));
  const long base = 2 * halfb + 1;
  const double rsquare = radius * radius;
  const long ntotal = powlong(base, ndim);

  long count = 0;

  // This is the counter
  std::vector<long> index(ndim, 0);

  // Loop over the total number of points to test, ticking over the
  // counter as we go.
  for (long n = 0; n < ntotal; ++n) {
    double rtestsq = 0;
    for (long k = 0; k < ndim; ++k) {
      double xk = index[k] - halfb;
      rtestsq += xk * xk;
    }
    if (rtestsq < rsquare) ++count;
    addone(index, base, 0);
  }
  return count;
}


int main(int argc, char* argv[]) 
{

    const double r = atof(argv[1]); 
    const long  nd = atol(argv[2]);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const long halfb = static_cast<long>(floor(r));
    const long base = 2 * halfb + 1;
    const long ntotal = powlong(base, nd);
    const double rsquare = r * r;

    unsigned long long int *d_count;
    unsigned long long int count;
    count=0 ;

    cudaMalloc(&d_count, sizeof(unsigned long long int));
    cudaMemcpy(d_count, &count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    unsigned long long int threadsPerBlock = 1024;
    unsigned long long int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock ;
    
    std::cout << "### " << " Radius " << r << "Dimension " << nd << " Total Points " << ntotal << std::endl;
    std::cout << "total threads " << " " << threadsPerBlock * blocksPerGrid<< " " << std::endl;
    cudaEventRecord(start, 0);
    count_in_v1_gpu<<<blocksPerGrid, threadsPerBlock>>>( ntotal, base, halfb, rsquare, nd, d_count );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;  // Must be a float
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  
    cudaMemcpy( &count, d_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    std::cout << " GPU -> " << count << std::endl;
    std::cout << "Kernel took: " << time << " ms" << std::endl;
    cudaFree(d_count);
    
    clock_t tstart = clock();
    const long num1 = count_in_v1(nd, r);
    clock_t tend = clock();
    double tms = diffclock(tend, tstart);
    std::cout << " CPU v1-> " << num1 << std::endl;
    std::cout << "# Time elapsed: " << tms << " ms " << std::endl;

    tstart = clock();
    const long num2 = count_in_v2(nd, r);
    tend = clock();
    tms = diffclock(tend, tstart);
    
    std::cout << " CPU v2-> " << num2 << std::endl;
    std::cout << "# Time elapsed: " << tms << " ms " << std::endl;

}

