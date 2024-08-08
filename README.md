# KMeans implemented from scratch using CUDA

### Performance
Tested on a T4 with K=3 and 100 2D datapoints.  
![CUDA KMeans Training Time Analysis](https://raw.githubusercontent.com/Tyler-Hilbert/CUDA-KMeans/36362c92e2a2f4f22d334d02b7655eeda30e886f/CUDA_Training.png)

### Total Runtime Data
| Phase          | Total Time (ms) |
|----------------|-----------------|
| Total Classify | 0.472           |
| Total Update   | 0.390           |

### Runtime Data by Epoch
| Phase        | Epoch | Time (ms) |
|--------------|-------|-----------|
| Classify     | 1     | 0.248     |
| Update       | 1     | 0.050     |
| Classify     | 2     | 0.026     |
| Update       | 2     | 0.036     |
| Classify     | 3     | 0.026     |
| Update       | 3     | 0.037     |
| Classify     | 4     | 0.024     |
| Update       | 4     | 0.037     |
| Classify     | 5     | 0.024     |
| Update       | 5     | 0.039     |
| Classify     | 6     | 0.023     |
| Update       | 6     | 0.039     |
| Classify     | 7     | 0.024     |
| Update       | 7     | 0.037     |
| Classify     | 8     | 0.024     |
| Update       | 8     | 0.039     |
| Classify     | 9     | 0.024     |
| Update       | 9     | 0.038     |
| Classify     | 10    | 0.022     |
| Update       | 10    | 0.039     |
