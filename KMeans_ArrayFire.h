// KMeans in ArrayFire for comparison. (Not currently included in benchmark / performance test).
// Reference (with modifications): https://arrayfire.org/docs/machine_learning_2kmeans_8cpp-example.htm

#ifndef __KMEANS_ARRAY_FIRE_H__
#define __KMEANS_ARRAY_FIRE_H__

#include <arrayfire.h>

using namespace af;


class KMeans_ArrayFire {
    public:
        
        KMeans_ArrayFire(float *data, int n, int d, int k);
        ~KMeans_ArrayFire();

        void one_epoch();
        void print_centroids();
        void print_predictions();
        
    private:
        // Array Fire
        array d_data;
        array d_centroids;

        // Data
        float *h_data; // Structure of Arrays. Memory stored in .cpp not class. Size: N*D

        // Dataset 
        int N; // Number of data elements (i. e. { x0, x1, y0, y1 } would be 2)
        int D; // Number of dimensions
        int K; // Number of clusters
};

#endif // __KMEANS_ARRAY_FIRE_H__