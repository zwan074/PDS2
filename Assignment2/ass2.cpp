#include <iostream>
#include "mpi.h"
#include <vector>
#include <stdio.h>
#include <math.h>

using namespace std;

/*
int empty_small(std::vector<float>& small_bucket,
    std::vector<int>& numpb,
    std::vector<float>& big_bucket, int num_data_pp)
    {
    int num_buckets = numpb.size();
    std::vector<int> recvcnt(num_buckets);
    MPI::COMM_WORLD.Alltoall(&numpb[0], 1, MPI_INT, &recvcnt[0], 1, MPI_INT);
    std::vector<int> recvoff(num_buckets);
    recvoff[0] = 0;
    int num_recv = recvcnt[0];
    for (int n = 1; n < num_buckets; ++n) {
        recvoff[n] = recvoff[n-1] + recvcnt[n-1];
        num_recv += recvcnt[n];
    }
    std::vector<int> sendoff(num_buckets);
    for (int n = 0; n < num_buckets; ++n) sendoff[n] = n * num_data_pp;
    MPI::COMM_WORLD.Alltoallv(
    &small_bucket[0], &numpb[0], &sendoff[0], MPI_FLOAT,
    &big_bucket[0], &recvcnt[0], &recvoff[0], MPI_FLOAT);
    return num_recv;
}*/


int main(int argc, char *argv[])
{
    const int ndata = 5;
    const float xmin = 10.0;
    const float xmax = 100.0;

    MPI::Init(argc, argv);
    int numproc = MPI::COMM_WORLD.Get_size(); // number of buckets
    int myid    = MPI::COMM_WORLD.Get_rank();

    // Fill up the array with data to send to the destination node. Note
    // that the contents of the array will
    int * sendbuf_rand_nums = new int[ndata * numproc];
    int * recvbuf_rand_nums = new int[ndata];
    
    float stepsize = (xmax - xmin) / numproc;

    int root = 0;

    if (myid == root) {
    cout << "SEND " << myid << " : "  << endl;;
    for (int i = 0; i < ndata * numproc; ++i) {
        sendbuf_rand_nums[i] = drand48()*(xmax-xmin-1)+xmin;
        //cout << sendbuf[i] << " ";
    }   
    

    MPI::COMM_WORLD.Scatter(sendbuf_rand_nums, ndata, MPI_FLOAT, recvbuf_rand_nums, ndata, MPI_FLOAT, root);

    vector<vector<float>> small_bucket;

    for (int i = 0; i < numproc; ++i) 
        small_bucket.push_back(new vector<float>);

    cout << "RECV " << myid << " : ";
    for (int i = 0; i < ndata; ++i) {
        int bktno = (int)floor((recvbuf_rand_nums[i] - xmin) / stepsize);
        small_bucket[bktno].push_back(recvbuf_rand_nums[i]);
    }
    cout << endl;

    for (int i = 0; i < ndata; ++i) {
        for (int j = 0; j < small_bucket[i].size(); ++j){
            cout << small_bucket[i][j] << "," ; 
        }
        cout << endl;
    }
    

    MPI::Finalize();

    delete[] sendbuf_rand_nums;
    delete[] recvbuf_rand_nums;
}
