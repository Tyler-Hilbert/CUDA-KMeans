# KMeans implemented from scratch using CUDA  

Tested with 1,000,000 2d data points with 3 clusters on T4.  

![CUDA KMeans Performance Test Table 1](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Performance/Table1.png)  
## Table 1: 'cuda_api_sum'  
| Name                  | Time (%) | Total Time (ms) | Num Calls | Avg (ms) | Med (ms) | Min (ms) | Max (ms) | StdDev (ms) |  
|-----------------------|----------|-----------------|-----------|----------|----------|----------|----------|-------------|  
| cudaMalloc            | 52.2     | 239,244.0       | 4         | 59,811.0 | 79.6     | 3.4      | 239,081.4 | 119,513.6   |  
| cudaLaunchKernel      | 31.8     | 145,860.1       | 20        | 7,293.0  | 7.8      | 4.6      | 145,661.3 | 32,568.5    |  
| cudaDeviceSynchronize | 15.2     | 69,665.9        | 20        | 3,483.3  | 3,174.0  | 5.0      | 7,044.0   | 3,570.9     |  
| cudaMemcpy            | 0.5      | 2,172.2         | 12        | 181.0    | 21.5     | 14.6     | 1,899.4   | 541.3       |  
| cudaMemset            | 0.2      | 1,023.8         | 20        | 51.2     | 3.3      | 2.1      | 955.4     | 212.8       |  
| cudaFree              | 0.1      | 618.7           | 4         | 154.7    | 128.5    | 4.2      | 357.5     | 174.7       |  
| cuModuleGetLoadingMode| 0.0      | 0.7             | 1         | 0.7      | 0.7      | 0.7      | 0.7       | 0.0         |  

![CUDA KMeans Performance Test Table 2](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Performance/Table2.png)  
## Table 2: 'cuda_gpu_kern_sum'  
| Kernel              | Time (%) | Total Time (ms) | Instances | Avg (ms) | Med (ms) | Min (ms) | Max (ms) | StdDev (ms) |  
|---------------------|----------|-----------------|-----------|----------|----------|----------|----------|-------------|  
| sum_and_count       | 100.0    | 69,530.1        | 10        | 6,953.0  | 7,021.0  | 6,340.9  | 7,021.1  | 215.0       |  
| update_centroids    | 0.0      | 33.1            | 10        | 3.3      | 3.3      | 3.3      | 3.4      | 34.4        |  

![CUDA KMeans Performance Test Table 3](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-LinearRegression/main/Performance/Table3.png)  
## Table 3: 'cuda_gpu_mem_time_sum'  
| Operation                         | Time (%) | Total Time (ms) | Count | Avg (ms) | Med (ms) | Min (ms) | Max (ms) | StdDev (ms) |  
|-----------------------------------|----------|-----------------|-------|----------|----------|----------|----------|-------------|  
| CUDA memcpy Host-to-Device        | 98.0     | 1,655.7         | 2     | 827.8    | 827.8    | 0.6      | 1,655.1  | 1,169.8     |  
| CUDA memcpy Device-to-Host        | 1.0      | 17.0            | 10    | 1.7      | 1.6      | 1.6      | 2.0      | 188.0       |  
| CUDA memset                       | 1.0      | 16.3            | 20    | 0.8      | 0.6      | 0.6      | 1.4      | 302.2       |  
