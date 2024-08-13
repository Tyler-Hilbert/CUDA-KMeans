# K-Means implemented from scratch using CUDA  
## Kernels
```C++
// Computes the sum (d_sum) and count (d_count) 
//    for each of the k clusters labeled in d_centroids.
// n: number of data points
// d: number of dimensions
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
);
```

```C++
// Updates each centroid using d_sum and d_count
//    where the index is d * centroid number (out of k).
// d: number of dimensions
// k: number of clusters
static __global__ void update_centroids(
    float *d_centroids,
    const float *d_sum,
    const int *d_count,
    int d,
    int k
);
```

## Performance
Tested with 1,000,000 2d data points with 3 clusters on T4.  
sum_and_count kernel was optimized from 69.5ms to 7.2ms by reducing threads per block from 256 to 32 and adding a shared memory.  

### Table 1: 'cuda_api_sum'  
 | Name                   |  Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |  
 |------------------------|-----------|-----------------|-----------|----------|----------|----------|----------|-------------|  
 | cudaMalloc             | 95.2      | 218,170,270     | 4 | 54,542,567.5 | 113,509.5 | 4,744 | 217,938,507 | 108,930,668.2 |  
 | cudaDeviceSynchronize  | 3.2       | 7,423,066       | 20 | 371,153.3 | 333,606.0 | 5,366 | 749,336 | 372,929.9 |  
 | cudaMemcpy             | 1.0       | 2,229,311       | 12 | 185,775.9 | 20,722.0 | 14,077 | 1,971,025 | 562,325.6 |  
 | cudaLaunchKernel       | 0.3       | 700,582         | 20 | 35,029.1 | 8,747.0 | 4,238 | 458,495 | 100,399.2 |  
 | cudaFree               | 0.2       | 539,348         | 4 | 134,837.0 | 75,913.0 | 3,798 | 383,724 | 178,359.7 |  
 | cudaMemset             | 0.0       | 86,019          | 20 | 4,300.9 | 3,493.5 | 1,988 | 9,747 | 2,022.5 |  
 | cuModuleGetLoadingMode | 0.0       | 950             | 1 | 950.0 | 950.0 | 950 | 950 | 0.0 |  

![CUDA KMeans Performance Test Table 1](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table1.png) 
 
### Table 2: 'cuda_gpu_kern_sum'  
 | Kernel           |  Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |  
 |------------------|-----------|-----------------|-----------|----------|----------|----------|----------|-------------|  
 | sum_and_count    | 99.5 | 7,276,289 | 10 | 727,628.9 | 738,047.5 | 637,651 | 744,656 | 32,161.5 |  
 | update_centroids | 0.5 | 35,134 | 10 | 3,513.4 | 3,488.0 | 3,487| 3,615 | 47.0 |  
 
![CUDA KMeans Performance Test Table 2](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table2.png) 

### Table 3: 'cuda_gpu_mem_time_sum'  
 | Operation                  |  Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) |  
 |----------------------------|-----------|-----------------|-------|----------|----------|----------|----------|-------------|  
 | CUDA memcpy Host-to-Device | 98.1 | 1,718,202 | 2 | 859,101.0 | 859,101.0 | 704 | 1,717,498 | 1,213,956.7 |  
 | CUDA memcpy Device-to-Host | 1.0 | 17,503 | 10 | 1,750.3 | 1,664.0 | 1,631 | 2,144 | 199.7 |  
 | CUDA memset                | 0.9 | 15,585 | 20 | 779.3 | 672.0 | 640 | 1,408 | 262.3 |  
 
![CUDA KMeans Performance Test Table 3](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table3.png) 

![CUDA KMeans Performance Test Table 4](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table4.png) 