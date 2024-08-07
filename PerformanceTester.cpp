// Performance tester for KMeans on CUDA

#include <stdio.h>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>

#include "KMeans_CUDA.cu"


#define EPOCHS 10   // Number of Epochs

// Generated Data Parameters
#define N 100       // Number of data points
#define D 2         // Dimensions of each point
#define K 3         // Number of clusters
// Generated Clusters
#define K0X 0.2
#define K0Y 0.2
#define K1X 0.8
#define K1Y 0.2
#define K2X 0.5 
#define K2Y 0.8

using namespace std;

// Generate random clustered data
// Data is in format { x0, y0, x1, y1, x2, y2... }
void generateClusteredData(float *data) {
    if (N != 100){
        printf ("error: N\n");
        return;
    }
    if (D != 2){
        printf ("error: D\n");
        return;
    }
    if (K != 3){
        printf ("error: K\n");
        return;
    }

    // Centroids
    vector< vector<float> > centroids = {
        {K0X, K0Y},
        {K1X, K1Y},
        {K2X, K2Y}
    };

    // Random number generator for Gaussian distribution
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 0.05); // Mean = 0, Stddev = 0.05

    // Generate points around each centroid
    int index = 0;
    for (const auto &centroid : centroids) {
        for (int i = 0; i < N/K; ++i) {
            for (int dim = 0; dim < D; ++dim) {
                data[index*D + dim] = centroid[dim] + distribution(generator);
            }
            ++index;
        }
    }

    // Handle the case where N is not exactly divisible by K
    int remaining_points= N - index;
    if (remaining_points> 0) {
        for (int i = 0; i < remaining_points; ++i) {
            for (int j = 0; j < D; ++j) {
                data[index * D + j] = centroids[0][j] + distribution(generator); // Add to the first centroid
            }
            ++index;
        }
    }
}

// Function to write data to a CSV file
bool writeDataToCSV(const vector<float> &data, const string &filename) {
    ofstream out_file(filename);

    // Check if the file opened successfully
    if (!out_file.is_open()) {
        printf ("error: Failed to open the file.\n");
        return false;
    }

    // Write the data
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            out_file << data[i * D + j];
            if (j < D - 1) out_file << ","; // Add a comma except after the last column
        }
        out_file << "\n"; // Newline for each data point
    }

    out_file.close();
    printf("Data successfully written to %s\n", filename.c_str());
    return true;
}


int main() {
    // Generate random data
    srand(static_cast<unsigned>(time(nullptr)));
    vector<float> data(N * D); // Data is in format { x0, y0, x1, y1, x2, y2... }
    generateClusteredData(data.data());
    writeDataToCSV(data, "kmeans_data.csv");

    // Timing variables
    chrono::time_point<chrono::system_clock> c_start;
    chrono::time_point<chrono::system_clock> c_end;
    vector<chrono::time_point<chrono::system_clock>> print_marks(EPOCHS);
    vector<chrono::time_point<chrono::system_clock>> classify_marks(EPOCHS);
    vector<chrono::time_point<chrono::system_clock>> update_marks(EPOCHS);

    // CUDA KMeans from scratch
    c_start = chrono::system_clock::now();
    KMeans_CUDA model(data.data(), N, D, K);
    CUDA_CHECK( cudaDeviceSynchronize() );
    c_end = chrono::system_clock::now();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf ("Epoch %i\n", epoch);
        model.printCentroids();
        CUDA_CHECK( cudaDeviceSynchronize() );
        print_marks.at(epoch) = chrono::system_clock::now();

        model.classify();
        CUDA_CHECK( cudaDeviceSynchronize() );
        classify_marks.at(epoch) = chrono::system_clock::now();
        
        model.update();
        CUDA_CHECK( cudaDeviceSynchronize() );
        update_marks.at(epoch) = chrono::system_clock::now();
    }
    model.printCentroids();

    // Print performance
    printf("Constructor:\t%ld ns\n", chrono::duration_cast<chrono::nanoseconds>(c_end - c_start).count());
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        printf("Classify:\t%ld ns\n", chrono::duration_cast<chrono::nanoseconds>(classify_marks.at(epoch) - print_marks.at(epoch)).count());
        printf("Update:\t\t%ld ns\n", chrono::duration_cast<chrono::nanoseconds>(update_marks.at(epoch) - classify_marks.at(epoch)).count());
    }
}