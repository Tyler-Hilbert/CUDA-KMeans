// KMeans in ArrayFire for comparison.
// Reference (with modifications): https://arrayfire.org/docs/machine_learning_2kmeans_8cpp-example.htm

#ifndef __KMEANS_ARRAY_FIRE_CPP__
#define __KMEANS_ARRAY_FIRE_CPP__

#include "KMeans_ArrayFire.h"

#include <stdio.h>
#include <random>

using namespace std;

KMeans_ArrayFire::KMeans_ArrayFire(
    float *data,
    int n,
    int d,
    int k
) {

    // CPU stack memory
    this->N = n;
    this->D = d;
    this->K = k;
    this->h_data = data;

    // Array Fire
    af::setBackend(AF_BACKEND_CUDA);
    setSeed(static_cast<unsigned long long>(time(0)));

    // Data
    d_data = array (N,  1,  D,  h_data);
    ///af_print(d_data);
    // Data Normalize
    ///printf ("Normalized\n");
    d_data = (d_data - min(d_data, 0)) / max(d_data, 0);
    ///af_print(d_data);

    // Initialize centroids from random data
    array random_indices = randu(K, u32) * N; // Generate K random indices in range [0, N)
    d_centroids = d_data(1, random_indices, span);  // Select rows from d_data
    ///af_print (d_centroids);
}



KMeans_ArrayFire::~KMeans_ArrayFire() {
    // Note don't delete h_data since it lives in Main.cpp
}



void KMeans_ArrayFire::print_centroids() {
    af_print(d_centroids);
}



void KMeans_ArrayFire::print_predictions() {
    // Data Broadcast
    array d_data_broadcast = tile(d_data, 1,  K,  1);
    // Centroids Broadcast
    array d_centroids_broadcast = tile(d_centroids, N,  1,  1);
    // Euclidean distance (squared)
    array d_distance = sum(pow(d_data_broadcast - d_centroids_broadcast, 2), 2);
    // Cluster
    array d_min_val, d_min_i;
    min(d_min_val, d_min_i, d_distance, 1);
    af_print(d_min_i);
}



// Runs one epoch of KMeans
void KMeans_ArrayFire::one_epoch() {
    // Broadcast
    array d_data_broadcast = tile(d_data, 1,  K,  1);
    array d_centroids_broadcast = tile(d_centroids, N,  1,  1);
    // Euclidean distance (squared)
    array d_distance = sum(pow(d_data_broadcast - d_centroids_broadcast, 2), 2);
    ///af_print (d_distance);

    // Cluster
    array d_min_val, d_min_i;
    min(d_min_val, d_min_i, d_distance, 1);
    ///af_print(d_min_i);

    // Update centroids
    gfor(seq ii, K) {
        d_centroids(span, ii, span) = sum(d_data*(d_min_i==ii)) / (sum(d_min_i==ii)+1e-5);
    }

    ///af_print (d_centroids);
}

#endif // __KMEANS_ARRAY_FIRE_CPP__