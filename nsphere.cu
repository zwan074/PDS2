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

const long MAXDIM = 10;
const double RMIN = 2.0;
const double RMAX = 8.0;

long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
  long p = 1;
  for (long i = 0; i < k; ++i) p *= n;
  return p;
}

/*----------------------------------------------------------------------------*/


__global__ void count_in_v1_gpu (long base, long halfb, double rsquare, long ndim , float* count )
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;

  
  // Indices in x,y,z,.... 

  long* index = (long*)malloc(ndim * sizeof(long));


  for (long i = 0; i < ndim; ++i) index[i] = 0;
  
  long idx = 0;
  while (n != 0) {
    long rem = n % base;
    n = n / base;
    index[idx] = rem;
    ++idx;
  }

  double rtestsq = 0;

  for (long k = 0; k < ndim; ++k) {
    double xk = index[k] - halfb;
    rtestsq += xk * xk;
  }
  if (rtestsq < rsquare) atomicAdd(count,1.0);

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
  // You can make this larger if you want
  const long ntrials = 20;


  for (long n = 0; n < ntrials; ++n) {

    
    // Get a random value for the hypersphere radius between the two limits
    const double r = drand48() * (RMAX - RMIN) + RMIN;

    // Get a random value for the number of dimensions between 1 and
    // MAXDIM inclusive
    const long  nd = lrand48() % (MAXDIM - 1) + 1;
    std::cout << "### " << n << " " << r << " " << nd << " ... " << std::endl;

    const long halfb = static_cast<long>(floor(r));
    const long base = 2 * halfb + 1;
    const long ntotal = powlong(base, nd);
    const double rsquare = r * r;

    float* h_count = (float*)malloc(sizeof(float));
    float *d_count;

    cudaMalloc(&d_count, sizeof(float));
    int threadsPerBlock = 256;
    int blocksPerGrid = ntotal / threadsPerBlock;

    count_in_v1_gpu<<<blocksPerGrid, threadsPerBlock>>>( base, halfb, rsquare, nd, d_count );
    cudaMemcpy( h_count, d_count, sizeof(float), cudaMemcpyDeviceToHost);

    
    //const long num1 = count_in_v1(nd, r);
    //const long num2 = count_in_v2(nd, r);
    std::cout << " -> " << h_count[0] << std::endl;
    cudaFree(d_count);
  }






}

