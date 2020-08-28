#include <iostream>
#include "mpi.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}


int main(int argc, char *argv[])
{
    const int ndata = 5000;
    const float xmin = 1.0;
    const float xmax = 100.0;
    double T0,T1,T2,T3;
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
        T0 = MPI_Wtime();
        //cout << "SEND " << myid << " : "  << endl;
        for (int i = 0; i < ndata * numproc; ++i) {
            sendbuf_rand_nums[i] = drand48()*(xmax-xmin-1)+xmin;
            //cout <<  sendbuf_rand_nums[i] << ",";
        }   
        //cout << endl;
        T0 = MPI_Wtime() - T0;
        cout << "SEND T0 " << myid << " : "  << T0 << endl;
    }
    
    T1 = MPI_Wtime();
    MPI::COMM_WORLD.Scatter(sendbuf_rand_nums, ndata, MPI_FLOAT, recvbuf_rand_nums, ndata, MPI_FLOAT, root);

    vector<vector<float> > small_bucket_2d;
    vector<float>  small_bucket_1d;
    vector<int> nitems;

    for (int i = 0; i < numproc; ++i) {
        vector<float> sub_small_bucket;
        small_bucket_2d.push_back(sub_small_bucket);
    }

    for (int i = 0; i < ndata; ++i) {
        int bktno = (int)floor((recvbuf_rand_nums[i] - xmin) / stepsize);
        small_bucket_2d[bktno].push_back(recvbuf_rand_nums[i]);
    }

    for (int i = 0; i < small_bucket_2d.size(); ++i) {
        nitems.push_back(small_bucket_2d[i].size());
        //cout << "SMALL BUCKET RECV " << myid << " : " << " bucket No. " << i << endl;
        //cout << "SMALL BUCKET ITEMS " << myid << " : " << nitems[i] << endl;
        for (int j = 0; j < small_bucket_2d[i].size() ; ++j){
            //cout << small_bucket_2d[i][j] << "," ; 
            small_bucket_1d.push_back(small_bucket_2d[i][j]);
        }
        //cout << endl;
    }
    //cout << "SMALL BUCKET 1D " << myid << " : " << endl;
    //for (int i = 0; i < small_bucket_1d.size() ; ++i){
        //cout << small_bucket_1d[i] << "," ; 
    //}
    //cout << endl;
    //T1 = MPI_Wtime() - T1;
    //cout << "SMALL BUCKET 1D T1 " << myid << " : " << T1 << endl;


    MPI_Barrier(MPI_COMM_WORLD);

    //T2 = MPI_Wtime();
    vector<int> recvcnt(numproc);
    MPI::COMM_WORLD.Alltoall(&nitems[0], 1, MPI_INT, &recvcnt[0], 1, MPI_INT);
    vector<int> recvoff(numproc);
    recvoff[0] = 0;
    //cout << "BIG BUCKET OFFSETS " << myid << endl;
    for (int n = 1; n < numproc; ++n) {
        recvoff[n] = recvoff[n-1] + recvcnt[n-1];
        //cout << "BIG BUCKET OFFSETS " << recvoff[n] << ",";
    }
    //cout << endl;

    int big_bucket_size = 0 ;
    for (int i = 0; i < recvcnt.size() ; i++) {
        big_bucket_size += recvcnt[i];
    }

    vector<float> big_bucket( big_bucket_size );
    vector<int> sendoff(numproc);
    sendoff[0] = 0;
    //cout << "SMALL BUCKET OFFSETS " << myid << endl;
    for (int n = 1; n < numproc; ++n) {
        sendoff[n] = sendoff[n-1] + nitems[n-1];
        //cout << "SMALL BUCKET OFFSETS " << sendoff[n] << ",";
    }
    //cout << endl;
    //T2 = MPI_Wtime() - T2;
    //cout << "SMALL BUCKET OFFSETS T2 " << myid << " : "  << T2 << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI::COMM_WORLD.Alltoallv(
        &small_bucket_1d[0], &nitems[0], &sendoff[0], MPI_FLOAT,
        &big_bucket[0], &recvcnt[0], &recvoff[0], MPI_FLOAT);
    
    qsort (&big_bucket[0], big_bucket.size(), sizeof(float), compare);

    //cout << "BIG BUCKET No. " << myid << endl;
    
    //for (int i = 0; i < big_bucket.size() ; i++) {
        //cout << big_bucket[i] << "," ;
    //}
    //cout << endl;

    MPI_Barrier(MPI_COMM_WORLD);

    vector<float> final_sorted_vector( ndata * numproc );
    vector<int> sendcnt_final(1);
    vector<int> recvcnt_final(numproc);
    sendcnt_final[0] = big_bucket.size();

    MPI::COMM_WORLD.Gather( &sendcnt_final[0], 1, MPI_INT, &recvcnt_final[0], 1, MPI_INT, root);

    vector<int> recvcnt_final_off(numproc);
    recvcnt_final_off[0] = 0;
    for (int n = 1; n < numproc; ++n) {
        recvcnt_final_off[n] = recvcnt_final_off[n-1] + recvcnt_final[n-1];
    }

    MPI::COMM_WORLD.Gatherv( &big_bucket[0], big_bucket.size(), MPI_FLOAT, 
                            &final_sorted_vector[0], &recvcnt_final[0], &recvcnt_final_off[0] , MPI_FLOAT, 0);

    T1 = MPI_Wtime() - T1;
    cout << "T1 " << myid << " : " << T1 << endl;

    if (myid == root) {
        cout << " final vector : " << final_sorted_vector.size() << endl;
        for (int i = 0; i < final_sorted_vector.size() ; i++) {
            cout << final_sorted_vector[i] << "," ;
        }
        cout << endl;
    }
    

    MPI::Finalize();

}
