# K-Means implemented from scratch using CUDA  
## Usage  
```C++
KMeans_CUDA model (data, N, D, K); // Where data is AoS, N is number of data points, D is number of dimensions and K is number of clusters
model.one_epoch(); // Trains one epoch
model.print_predictions(); // Prints the classifications. Can be commented out.
// printf ("Error: %f\n", model.compute_error()); // Uncomment to print error
```  

## Kernels
```C++
// Updates each centroid using d_sum and d_count
//    where the index is d * centroid number (out of k).
// d: number of dimensions
// k: number of clusters
__global__ void update_centroids(
    float *d_centroids,
    const float *d_sum,
    const int *d_count,
    int d,
    int k
);
```  

```C++
// Computes the sum (d_sum) and count (d_count) 
//    for each of the k clusters labeled in d_centroids.
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
);
```  

```C++
// Computes error and updates d_error
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
```  

## Performance  
Go to: https://github.com/Tyler-Hilbert/CUDA-KMeans/tree/52db75728794449dc152989c648e03b632d24c08 for most recent performance tests.  

## Performance compared to scikit-learn and ArrayFire  
It is shown that this implementations of K-Means outperforms scitkit-learn and ArrayFire on a T4.  
![CUDA KMeans Performance vs scikit-learn and ArrayFire](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/main/Performance/Comparison.png)  

## Usage Continued  
To compile the test program:  
$git clone https://github.com/Tyler-Hilbert/CUDA-KMeans.git  
$cd CUDA-KMeans  
$nvcc main.cpp KMeans_CUDA.cu -o kmeans  
$./kmeans  