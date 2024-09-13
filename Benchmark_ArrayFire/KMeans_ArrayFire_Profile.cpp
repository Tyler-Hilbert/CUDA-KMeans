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

    af::sync();

    // Data
    d_data = array (N,  1,  D,  h_data);

    af::sync();

    // Data Normalize
    d_data = (d_data - min(d_data, 0)) / max(d_data, 0);

    af::sync();

    // Initialize centroids from random data from dataset
    array random_indices = floor(randu(K) * N).as(u32);
    af::sync();
    d_centroids = d_data(random_indices, span);
    af::sync();
    d_centroids = moddims(d_centroids, 1, K, D, 1);
    af::sync();
}



KMeans_ArrayFire::~KMeans_ArrayFire() {
    // Note don't delete h_data since it lives in Main.cpp
}



void KMeans_ArrayFire::print_centroids() {
    //af_print(d_centroids);
}



void KMeans_ArrayFire::print_predictions() {
    af::sync();

    // Data Broadcast
    array d_data_broadcast = tile(d_data, 1,  K,  1);

    af::sync();

    // Centroids Broadcast
    array d_centroids_broadcast = tile(d_centroids, N,  1,  1);

    af::sync();

    // Euclidean distance (squared)
    array d_distance = sum(pow(d_data_broadcast - d_centroids_broadcast, 2), 2);

    af::sync();

    // Cluster
    array d_min_val, d_min_i;

    af::sync();

    min(d_min_val, d_min_i, d_distance, 1);

    af::sync();
}



// Runs one epoch of KMeans
void KMeans_ArrayFire::one_epoch() {

    af::sync();

    // Broadcast data and centroids to NxKxD size
    array d_data_broadcast = tile(d_data, 1,  K,  1);

    af::sync();

    array d_centroids_broadcast = tile(d_centroids, N,  1,  1);

    af::sync();

    // Calculate Euclidean distance (squared)
    array d_distance = sum(pow(d_data_broadcast - d_centroids_broadcast, 2), 2);

    af::sync();

    // Cluster
    array d_min_val, d_min_i;

    af::sync();

    min(d_min_val, d_min_i, d_distance, 1);

    af::sync();

    // Update centroids
    gfor(seq ii, K) {
        d_centroids(span, ii, span) = sum(d_data*(d_min_i==ii)) / (sum(d_min_i==ii)+1e-5);
    }

    af::sync();
}

#endif // __KMEANS_ARRAY_FIRE_CPP__