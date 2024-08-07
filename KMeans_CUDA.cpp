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


// Kernel to classify each point to the nearest centroid
static __global__ void classifyPoints(
    const float *d_data, 
    const float *d_centroids, 
    int *d_classes, 
    const int n, 
    const int d, 
    const int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // x and y
        float x = d_data[d*idx];
        float y = d_data[d*idx+1];
        
        // Find closest point
        int min_class = 0;
        float min_dist = sqrtf(
            (x - d_centroids[0]) * (x - d_centroids[0]) + (y - d_centroids[1]) * (y - d_centroids[1])
        );
        for (int c = 1; c < k; c++) {
            float dist = sqrtf(
                (x - d_centroids[c*d]) * (x - d_centroids[c*d]) + (y - d_centroids[c*d+1]) * (y - d_centroids[c*d+1])
            );

            if (dist < min_dist) {
                min_dist = dist;
                min_class = c;
            }
        }

        // Update
        d_classes[idx] = min_class;
    }
}

// Kernel to update centroids
static __global__ void updateCentroids(
    const float *d_data, 
    float *d_centroids, 
    const int *d_classes, 
    const int n,
    const int d,
    const int k
) {

    extern __shared__ float shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    const int COUNT = d * k; // The location in shared memory where the count starts

    // Calculate the sum of each x and y, followed by the count.
    // shared_data 0 to 2*k even is x, odd is y, and 2*k to 3*k is counts
    if (idx < n) {
        int c = d_classes[idx];
        atomicAdd(&shared_data[c*d], d_data[idx*d]);
        atomicAdd(&shared_data[c*d+1], d_data[idx*d+1]);
        atomicAdd(&shared_data[c+COUNT], 1);
    }
    __syncthreads();

    // Calculate mean
    if (tid < k) {
        if (shared_data[tid+COUNT] > 0) {
            d_centroids[tid*d] = shared_data[tid*d] / shared_data[tid+COUNT];
            d_centroids[tid*d+1] = shared_data[tid*d+1] / shared_data[tid+COUNT];
        }
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
            can_update = false;
            this->h_data = data;

            // CPU heap memory
            h_classes = new int[n];

            h_centroids = new float[d*k];
            for (int i = 0; i < d*k; i++) {
                h_centroids[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }

            // GPU memory
            CUDA_CHECK( cudaMalloc(&d_data, n*d*sizeof(float)) );
            CUDA_CHECK( cudaMalloc(&d_classes, n*sizeof(int)) );
            CUDA_CHECK( cudaMalloc(&d_centroids, k*d*sizeof(float)) );

            CUDA_CHECK( cudaMemcpy(d_data, h_data, n*d*sizeof(float), cudaMemcpyHostToDevice) );
            CUDA_CHECK( cudaMemcpy(d_centroids, h_centroids, k*d*sizeof(float), cudaMemcpyHostToDevice) );

            // GPU threads, blocks, shared memory
            threads_per_block = 256;
            blocks = (n + threads_per_block - 1) / threads_per_block;
            shared_memory_size = 3 * k * sizeof(float); // 3 for sum of x, y and count
        }

        ~KMeans_CUDA() {
            delete[] h_classes;
            delete[] h_centroids;

            CUDA_CHECK( cudaFree(d_data) );
            CUDA_CHECK( cudaFree(d_classes) );
            CUDA_CHECK( cudaFree(d_centroids) );
        }

        // Classifies each element
        void classify() {
            if (can_update) {
                return; // Classifications already up to date
            }

            // Run classification kernel
            classifyPoints<<<blocks, threads_per_block>>>(d_data, d_centroids, d_classes, n, d, k);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
            // Copy memory to host
            CUDA_CHECK( cudaMemcpy(h_classes, d_classes, n*sizeof(int), cudaMemcpyDeviceToHost) );

            can_update = true;
        }

        // Updates centroids
        bool update() {
            if (!can_update) {
                return false; // Need to classify before can update
            }

            updateCentroids<<<blocks, threads_per_block, shared_memory_size>>>(d_data, d_centroids, d_classes, n, d, k);
            CUDA_CHECK( cudaPeekAtLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() );
             // Copy memory to host
            CUDA_CHECK( cudaMemcpy(h_centroids, d_centroids, k*d*sizeof(float), cudaMemcpyDeviceToHost) );

            can_update = false;
            return true;
        }

        void printCentroids() {
            for (int i = 0; i < k; i++) {
                printf ("x %f  y %f\n", h_centroids[i*d], h_centroids[i*d+1]);
            }
        }


    private:
        // Data
        float *h_data; // Data is in format { x0, y0, x1, y1, x2, y2... }. Memory stored in .cpp not class
        float *d_data; // Pointer to data on GPU

        // Learned centroids
        float *h_centroids; // Pointer to centroids in format { c0x, c0y, c1x, c1y, c2x, c2y } on heap
        float *d_centroids; // Pointer to centroids on GPU

        // Classifications (inference)
        int *h_classes;
        int *d_classes;

        // Threads, blocks and shared memory size
        int threads_per_block;
        int blocks;
        int shared_memory_size;

        bool can_update; // Classifications have been made

        // Dataset 
        int n; // Number of data elements (i. e. { x0, y0, x1, y1, x2, y2} would be 3)
        int d; // Number of dimensions
        int k; // Number of clusters
};

#endif // __KMEANS_CU__