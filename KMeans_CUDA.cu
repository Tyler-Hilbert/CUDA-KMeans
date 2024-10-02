// KMeans in CUDA

#ifndef __KMEANS_CU__
#define __KMEANS_CU__

#include "KMeans_CUDA.h"

#include <stdio.h>
#include <random>
#include <ctime>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <float.h>

using namespace std;



#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



// Computes the sum (d_sum) and count (d_count) for each of the k clusters labeled in d_centroids.
// n: number of data points
// d: number of dimensions
// k: number of clusters
// Uses shared memory of 3*k*d
__global__ void sum_and_count(
    const float *d_data,
    const float *d_centroids,
    float *d_sum,
    int *d_count,
    int n,
    int d,
    int k
) {
    // Shared memory: 
    //   0 to k*d: centroids,
    //   k*d to 2*k*d: sum,
    //   2*k*d to 3*k*d: count
    extern __shared__ float s_shared[];
    float *s_centroids = s_shared;       // Shared memory for centroids
    float *s_sum = &s_centroids[k*d];    // Shared memory for sum
    float *s_count = &s_sum[k*d];       // Shared memory for count

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    if (tid < k * d) {
        s_centroids[tid] = d_centroids[tid];
    }
    if (tid < k) {
        s_count[tid] = 0;
    }
    if (tid < k * d) {
        s_sum[tid] = 0.0f;
    }
    __syncthreads(); // Ensure all shared memory is initialized

    if (idx < n) {
        const int idxd = idx * d;

        // Find closest centroid
        int min_class = -1;
        float dist;
        float min_dist = FLT_MAX;
        for (int c = 0; c < k; c++) {
            dist = 0;
            for (int i = 0; i < d; i++) {
                dist += pow(d_data[i+idxd] - s_centroids[i+c*d], 2);
            }
            if (dist < min_dist) {
                min_dist = dist;
                min_class = c;
            }
        }

        // Update sum and count
        atomicAdd(&s_count[min_class], 1);
        for (int i = 0; i < d; i++) {
            atomicAdd(&s_sum[i+min_class*d], d_data[i+idxd]);
        }
    }
    __syncthreads(); // Ensure all threads have finished updating sum and count

    // Write shared memory results to global memory (only one thread per centroid)
    if (tid < k) {
        atomicAdd(&d_count[tid], (int)s_count[tid]);
    }
    if (tid < k * d) {
        atomicAdd(&d_sum[tid], s_sum[tid]);
    }
}



// Updates each centroid using d_sum and d_count where the index is d * centroid number (out of K).
// d: number of dimensions
// k: number of clusters
__global__ void update_centroids(
    float *d_centroids,
    const float *d_sum,
    const int *d_count,
    int d,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update centroids
    if (idx < k) {
        const int idxd = idx * d;
        for (int i = 0; i < d; i++) {
          if (d_count[idx] != 0) {
            d_centroids[i+idxd] = d_sum[i+idxd] / d_count[idx];
          }
        }
    }
}



// Computes error and updates d_error
// Note this is currently only for debugging purposes as it recomputes work done in sum_and_count
// n: number of data points
// d: number of dimensions
// k: number of clusters
__global__ void calculate_error(
    const float *d_data,
    const float *d_centroids,
    float *d_error,
    int n,
    int d,
    int k
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        const int idxd = idx * d;

        // Find closest centroid
        float dist;
        float min_dist = FLT_MAX;
        for (int c = 0; c < k; c++) {
            dist = 0;
            for (int i = 0; i < d; i++) {
                dist += pow(d_data[i+idxd] - d_centroids[i+c*d], 2);
            }
            if (dist < min_dist) {
                min_dist = dist;
            }
        }

        // Add error
        atomicAdd(d_error, min_dist);
    }
}



KMeans_CUDA::KMeans_CUDA(
    float *data,
    int n,
    int d,
    int k
) {
    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // CPU stack memory
    this->n = n;
    this->d = d;
    this->k = k;

    // CPU heap memory
    // Normalize data
    float *mins = new float[d];
    float *maxs = new float[d];
    for (int i = 0; i < d; i++) {
        mins[i] = data[i];
        maxs[i] = data[i];
    }
    for (int i = 0; i < n*d; i++) {
        mins[i%d] = min(mins[i%d], data[i]);
        maxs[i%d] = max(maxs[i%d], data[i]);
    }
    /*
    // Print min and max for debug
    for (int i = 0; i < d; i++) {
        printf ("min %f, max %f\n", mins[i], maxs[i]);
    }
    */
    h_data = new float[n*d];
    for (int i = 0; i < n*d; i++) {
        h_data[i] = (data[i] - mins[i%d]) / maxs[i%d];
    }
    delete[] mins;
    delete[] maxs;
    /*
    // Print data for debug
    for (int i = 0; i < n*d; i++) {
        printf ("%f ", h_data[i]);
    }
    printf ("\n");
    */

    // Centroids
    h_centroids = new float[d*k];
    for (int i = 0; i < k; i++) { // Select a datapoint for each centroid initalization
        int data_index = rand() % n;
        for (int j = 0; j < d; j++) { // Select a dimension
            h_centroids[i*d+j] = h_data[data_index*d+j];
        }
    }

    // GPU memory
    CUDA_CHECK( cudaMalloc(&d_data,         n*d*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_centroids,    k*d*sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_count,        k*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&d_sum,          k*d*sizeof(float)) );

    CUDA_CHECK( cudaMemcpy(d_data,      h_data,       n*d*sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_centroids, h_centroids,  k*d*sizeof(float), cudaMemcpyHostToDevice) );
}



KMeans_CUDA::~KMeans_CUDA() {

    delete[] h_data;
    delete[] h_centroids;

    CUDA_CHECK( cudaFree(d_data) );
    CUDA_CHECK( cudaFree(d_centroids) );
    CUDA_CHECK( cudaFree(d_count) );
    CUDA_CHECK( cudaFree(d_sum) );
}



// Prints out the centroids
void KMeans_CUDA::print_centroids() {
    for (int i = 0; i < k; i++) {
        string s = "";
        for (int j = 0; j < d; j++) {
            s += to_string(h_centroids[i*d + j]);
            s += "  ";
        }
        s += "\n";
        printf (s.c_str());
    }
}



// Prints out predictions
void KMeans_CUDA::print_predictions() {
    for (int p = 0; p < n; p++) {
        // Find closest centroid
        int min_class = 0;
        float dist = 0;
        for (int i = 0; i < d; i++) {
            dist += pow(h_data[i+p*d] - h_centroids[i], 2);
        }

        float min_dist = dist;
        for (int c = 1; c < k; c++) {
            dist = 0;
            for (int i = 0; i < d; i++) {
                dist += pow(h_data[i+p*d] - h_centroids[i+c*d], 2);
            }
            if (dist < min_dist) {
                min_dist = dist;
                min_class = c;
            }
        }

        printf ("%i ", min_class);
    }
    printf ("\n");
}



// Runs one epoch of KMeans
void KMeans_CUDA::one_epoch() {
    // GPU setup
    CUDA_CHECK( cudaMemset(d_count, 0, k*sizeof(int)) );
    CUDA_CHECK( cudaMemset(d_sum,   0, k*d*sizeof(float)) );
    int threads_per_block = 32;
    int blocks1 = (n + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = 3*k*d*sizeof(float);

    // Run kernel to get sums and counts
    sum_and_count<<<blocks1, threads_per_block, shared_mem_size>>>(d_data, d_centroids, d_sum, d_count, n, d, k);
    CUDA_CHECK( cudaPeekAtLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Run kernel to update centroids (calculate average)
    int blocks2 = (k + threads_per_block - 1) / threads_per_block;
    update_centroids<<<blocks2, threads_per_block>>>(d_centroids, d_sum, d_count, d, k);
    CUDA_CHECK( cudaPeekAtLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy data back to host
    CUDA_CHECK( cudaMemcpy(h_centroids, d_centroids, k*d*sizeof(float), cudaMemcpyDeviceToHost) );
}



float KMeans_CUDA::compute_error() {
    printf ("Note: The compute error function isn't optimized currently and is only used for debugging\n");

    // GPU setup
    float *d_error;
    float h_error = 0;
    CUDA_CHECK( cudaMalloc(&d_error, sizeof(float)) );
    CUDA_CHECK( cudaMemset(d_error, 0, sizeof(float)) );

    int threads_per_block = 32;
    int blocks1 = (n + threads_per_block - 1) / threads_per_block;

    // Run kernel to get sums and counts
    calculate_error<<<blocks1, threads_per_block>>>(d_data, d_centroids, d_error, n, d, k);
    CUDA_CHECK( cudaPeekAtLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    // Copy data back to host
    CUDA_CHECK( cudaMemcpy(&h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost) );

    // Free memory
    CUDA_CHECK( cudaFree(d_error) );

    return h_error;
}



#endif // __KMEANS_CU__