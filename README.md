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

### Table 1: 'cuda_api_sum'  
 |  Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |  
 |-----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
FIXME
 
 ![CUDA KMeans Performance Test Table 1]()  
 
### Table 2: 'cuda_gpu_kern_sum'  
 |  Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |  
 |-----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|  
FIXME
 
![CUDA KMeans Performance Test Table 2]()  

### Table 3: 'cuda_gpu_mem_time_sum'  
 |  Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Operation |  
 |-----------|-----------------|-------|----------|----------|----------|----------|-------------|-----------|  
FIXME
 
![CUDA KMeans Performance Test Table 3]()  
