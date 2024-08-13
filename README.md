# KMeans implemented from scratch using CUDA  
## Kernels
```C++
// Computes the sum (d_sum) and count (d_count) 
//    for each of the k clusters labeled in d_centroids.
// n: number of data points
// d: number of dimensions
// k: number of clusters
static __global__ void sum_and_count(
    const float *d_data,
    const float *d_centroids,
    float *d_sum,
    int *d_count,
    int n,
    int d,
    int k
)
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

### Table 1: 'cuda_api_sum'  
 |  Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |  
 |-----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
 | 52.2 | 239,244,012 | 4 | 59,811,003.0 | 79,581.0 | 3,439 | 239,081,411 | 119,513,626.2 | cudaMalloc |  
 | 31.8 | 145,860,061 | 20 | 7,293,003.0 | 7,810.5 | 4,625 | 145,661,321 | 32,568,523.9 | cudaLaunchKernel |  
 | 15.2 | 69,665,917 | 20 | 3,483,295.9 | 3,173,981.0 | 5,015 | 7,044,049 | 3,570,872.7 | cudaDeviceSynchronize  |  
 | 0.5 | 2,172,189 | 12 | 181,015.8 | 21,476.5 | 14,635 | 1,899,442 | 541,282.9 | cudaMemcpy |  
 | 0.2 | 1,023,776 | 20 | 51,188.8 | 3,331.0 | 2,131 | 955,446 | 212,845.6 | cudaMemset |  
 | 0.1 | 618,662 | 4 | 154,665.5 | 128,489.5 | 4,173 | 357,510 | 174,656.6 | cudaFree |  
 | 0.0 | 724 | 1 | 724.0 | 724.0 | 724 | 724 | 0.0 | cuModuleGetLoadingMode |  
 
 ![CUDA KMeans Performance Test Table 1](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table1.png)  
 
### Table 2: 'cuda_gpu_kern_sum'  
 |  Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |  
 |-----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|  
 | 100.0 | 69,530,116 | 10 | 6,953,011.6 | 7,021,011.0 | 6,340,969 | 7,021,123 | 215,049.9 | sum_and_count |  
 | 0.0 | 33,087 | 10 | 3,308.7 | 3,296.0 | 3,264 | 3,392 | 34.4 | update_centroids |  
 
![CUDA KMeans Performance Test Table 2](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table2.png)  

### Table 3: 'cuda_gpu_mem_time_sum'  
 |  Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation |  
 |-----------|-----------------|-------|----------|----------|----------|----------|-------------|-----------|  
 | 98.0 | 1,655,697 | 2 | 827,848.5 | 827,848.5 | 640 | 1,655,057 | 1,169,849.5 | [CUDA memcpy Host-to-Device] |  
 | 1.0 | 16,993 | 10 | 1,699.3 | 1,632.0 | 1,568 | 2,048 | 188.0 | [CUDA memcpy Device-to-Host] |  
 | 1.0 | 16,321 | 20 | 816.0 | 640.0 | 608 | 1,376 | 302.2 | [CUDA memset] |  
 
![CUDA KMeans Performance Test Table 3](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Table3.png)  
