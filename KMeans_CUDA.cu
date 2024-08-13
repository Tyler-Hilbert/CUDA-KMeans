// KMeans in CUDA

#ifndef __KMEANS_CU__
#define __KMEANS_CU__

#include <stdio.h>
#include <random>
#include <stdexcept>

#include <cuda_runtime.h>


#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;


// Computes the sum (d_sum) and count (d_count) for each of the k clusters labeled in d_centroids.
// n: number of data points
// d: number of dimensions (should be 2)
// k: number of clusters
// Uses shared memory of 3*k*d
static __global__ void sum_and_count(
    const float *d_data,
    const float *d_centroids,
    float *d_sum,
    int *d_count,
    int n,
    int d,
    int k
) {
    // Shared memory: 0 to k*d: centroids, k*d to 2*k*d: sum, 2*k*d to 3*k*d: count
    extern __shared__ float s_shared[];
    float *s_centroids = s_shared;      // Shared memory for centroids
    float *s_sum = &s_shared[k*d];    // Shared memory for sum
    int *s_count = (int*)&s_sum[k*d]; // Shared memory for count

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
        // x and y
        const int idxd = idx * d;
        float x = d_data[idxd];
        float y = d_data[idxd + 1];

        // Find closest centroid
        int min_class = 0;
        float min_dist = abs(x - s_centroids[0]) + abs(y - s_centroids[1]);
        for (int c = 1; c < k; c++) {
            const int cd = c * d;
            float dist = abs(x - s_centroids[cd]) + abs(y - s_centroids[cd + 1]);
            if (dist < min_dist) {
                min_dist = dist;
                min_class = c;
            }
        }

        // Update sum and count
        int min_class_d = min_class * d;
        atomicAdd(&s_count[min_class], 1);
        atomicAdd(&s_sum[min_class_d], x);
        atomicAdd(&s_sum[min_class_d + 1], y);
    }
    __syncthreads(); // Ensure all threads have finished updating sum and count

    // Write shared memory results to global memory (only one thread per centroid)
    if (tid < k) {
        atomicAdd(&d_count[tid], s_count[tid]);
    }
    if (tid < k * d) {
        atomicAdd(&d_sum[tid], s_sum[tid]);
    }
}


// Updates each centroid using d_sum and d_count where the index is d * centroid number (out of K).
// d: number of dimensions (should be 2)
// k: number of clusters
static __global__ void update_centroids(
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
        d_centroids[idxd] = d_sum[idxd] / d_count[idx];
        d_centroids[idxd+1] = d_sum[idxd+1] / d_count[idx];
    }
}

class KMeans_CUDA {
    public:
        
        KMeans_CUDA(
            float *data,
            int n, 
            int d,
            int k
        ) {

            if (d != 2) {
                throw invalid_argument("Invalid d");
            }
            if (k != 3) {
                throw invalid_argument("Invalid k");
            }

            // CPU stack memory
            this->n = n;
            this->d = d;
            this->k = k;
            this->h_data = data;

            // CPU heap memory
            h_centroids = new float[d*k];
            for (int i = 0; i < d*k; i++) {
                h_centroids[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            // GPU memory
            CUDA_CHECK( cudaMalloc(&d_data,         n*d*sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&d_centroids,    k*d*sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&d_count,        k*sizeof(int)) );
            CUDA_CHECK( cudaMalloc(&d_sum,          k*d*sizeof(float)) );

            CUDA_CHECK( cudaMemcpy(d_data, h_data, n*d*sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_centroids, h_centroids, k*d*sizeof(float), cudaMemcpyHostToDevice) );


        }

        ~KMeans_CUDA() {
            // Note don't delete h_data since it lives in Main.cpp
            delete[] h_centroids;

            CUDA_CHECK( cudaFree(d_data) );
            CUDA_CHECK( cudaFree(d_centroids) );
            CUDA_CHECK( cudaFree(d_count) );
            CUDA_CHECK( cudaFree(d_sum) );
        }

        void printCentroids() {
            for (int i = 0; i < k; i++) {
                printf ("x %f  y %f\n", h_centroids[i*d], h_centroids[i*d+1]);
            }
        }

        // Runs one epoch of KMeans
        void one_epoch() {
            // GPU setup
            CUDA_CHECK( cudaMemset(d_count, 0, k*sizeof(int)) );
            CUDA_CHECK( cudaMemset(d_sum, 0, k*d*sizeof(int)) );
            int threads_per_block = 256;
            int blocks1 = (n + threads_per_block - 1) / threads_per_block;
            size_t shared_mem_size = (2*k*d*sizeof(float)) + (k*sizeof(int));

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
            CUDA_CHECK( cudaMemcpy(h_centroids, d_centroids, k*d*sizeof(int), cudaMemcpyDeviceToHost) );
        }

    private:
        // Data
        float *h_data; // Data is in format { x0, y0, x1, y1, x2, y2... }. Memory stored in .cpp not class. Size n*d
        float *d_data; // Pointer to data on GPU. Size n*d

        // Learned centroids
        float *h_centroids; // Pointer to centroids in format { c0x, c0y, c1x, c1y, c2x, c2y } on heap. Size k*d
        float *d_centroids; // Pointer to centroids on GPU. Size k*d

        // Count and Sum
        float *d_sum; // Size k*d
        int *d_count; // Size k

        // Dataset 
        int n; // Number of data elements (i. e. { x0, y0, x1, y1, x2, y2} would be 3)
        int d; // Number of dimensions
        int k; // Number of clusters
};

#endif // __KMEANS_CU__