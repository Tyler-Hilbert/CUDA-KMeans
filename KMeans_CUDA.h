// KMeans in CUDA

#ifndef __KMEANS_CU_H__
#define __KMEANS_CU_H_

class KMeans_CUDA {
    public:
        KMeans_CUDA(float *data, int n, int d, int k);
        ~KMeans_CUDA();

        void one_epoch();
        void print_centroids();
        void print_predictions();

    private:
        // Data
        float *h_data; // Data is in format Memory stored in .cpp not class. Size n*d
        float *d_data; // Pointer to data on GPU. Size n*d

        // Learned centroids
        float *h_centroids; // Pointer to centroids on heap. Size k*d
        float *d_centroids; // Pointer to centroids on GPU. Size k*d

        // Count and Sum
        float *d_sum; // Size k*d
        int *d_count; // Size k

        // Dataset 
        int n; // Number of data elements
        int d; // Number of dimensions
        int k; // Number of clusters
};

#endif // __KMEANS_CU_H__