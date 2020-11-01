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
#include <cuda.h>
#include <vector>
#include <fstream>
using namespace std;

const long MAXDIM = 8;
const double RMIN = 2.0;
const double RMAX = 8.0;
const int MAX_POINTS_PER_THREAD = 500;


double diffclock(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1 - clock2;
	double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
	return diffms; // Time difference in milliseconds
}

/*
 * Evaluate n**k where both are long integers
 */
long powlong(long n, long k)
{
	long p = 1;
	for (long i = 0; i < k; ++i) p *= n;
	return p;
}

/*
 * Convert a decimal number into another base system - the individual
 * digits in the new base are stored in the index array.
 */
void convert(long num, long base, std::vector<long>& index)
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
{
	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	const double rsquare = radius * radius;
	const long ntotal = powlong(base, ndim);
	cout << "ntotal:"<<ntotal<<endl;
	long count = 0;
	std::vector<long> index(ndim, 0);
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

// kernel
__global__ void cuda_func_count(int ndim, double radius, long nfrom, long nto, long nthreads, int* counter)
{
	long id = blockIdx.x * blockDim.x + threadIdx.x;
	counter[id] = 0;
	if (id >= nto)
		return;

	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	const double rsquare = radius*radius;
	
	long index = 0;
	long num = nfrom + id;
	while (num < nto)
	{	
		/*
		double rtestsq = 0;
		
		for (int i=0; i<ndim; i++)
		{
			long rem = num % base;
			num = num / base;
			double xk = rem - halfb;
			rtestsq += xk * xk;
		}*/
		
		long idx = 0;
		double rtestsq = 0;

		while (n != 0) {
			long rem = n % base;
			n = n / base;
			double xk = rem - halfb;
			rtestsq += xk * xk;
			++idx;
		}

		for (long k = idx; k < ndim; ++k) {
			double xk = 0.0 - halfb;
			rtestsq += xk * xk;
		}
		if (rtestsq < rsquare )
		{
			atomicAdd(&counter[id], 1);
		}

		
		
		index++;
		num = nfrom + id + nthreads*index;	
	}
}

long count_in_gpu(long ndim, double radius)
{
	const long halfb = static_cast<long>(floor(radius));
	const long base = 2 * halfb + 1;
	const long ntotal = powlong(base, ndim);
	cout << "ntotal:"<<ntotal<<endl;

	const int threadsPerBlock_x = (ntotal<1024)?ntotal:1024;

	int blocksPerGrid = ntotal / 1024 + 1;
	if (blocksPerGrid >  1024)
	{
		blocksPerGrid = 1024;
	}
	const long nthreads = threadsPerBlock_x*blocksPerGrid; //maximum 1024*1024 threads
	int* counters = new int[nthreads];
	memset(counters, 0, sizeof(int)*nthreads);
	int* d_counters;
	cudaMalloc(&d_counters, sizeof(int)*nthreads);

	long total_count = 0;
	//invoke the kernel
	//std::cout << "Launching a grid of " << nthreads << " threads" << std::endl;
	const long points_for_each = MAX_POINTS_PER_THREAD * nthreads;
	long nfrom = 0; 
	long nto = points_for_each;
	do
	{
		if (nto > ntotal){
			nto = ntotal;
		}
		cuda_func_count <<<blocksPerGrid, threadsPerBlock_x>>>(ndim, radius, nfrom, nto, nthreads, d_counters); 
		
		//copyback the counters to host
		cudaMemcpy(counters, d_counters, sizeof(int)*nthreads, cudaMemcpyDeviceToHost);
		//coculate all counters amount
		for (long i = 0; i < nthreads; i++)
		{
			total_count += counters[i];
		}

		nfrom = nto;
		nto += points_for_each;
	}while (nfrom < ntotal);

	cudaFree(d_counters);
	delete[] counters;

	return total_count;
}

int main(int argc, char* argv[])
{
  
  const double r = atof(argv[1]); 
  const long  nd = atol(argv[2]);
  
  clock_t tstart = clock();
  // const long seq_count = count_in_v1(nd, r);
  // double seq_t_cost = diffclock(clock(), tstart);
  // tstart = clock();

  /**********************************************/		
  const long cuda_count = count_in_gpu(nd, r);
  double cuda_t_cost = diffclock(clock(), tstart);
  
  cout <<"r:"<< r << "; nd:" << nd 
  // << "; seq_count:" << seq_count << "; seq_t_cost:" << seq_t_cost 
  <<"; cuda_count:"<< cuda_count << "; cuda_t_cost"<< cuda_t_cost <<endl;	

	return 0;

}

