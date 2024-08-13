// Performance tester for KMeans on CUDA

#include <stdio.h>
#include <vector>
#include <random>
#include <fstream>

#include "KMeans_CUDA.cu"


#define EPOCHS 10   // Number of Epochs

// Generated Data Parameters
#define N 1000000    // Number of data points
#define D 2         // Dimensions of each point
#define K 3         // Number of clusters
// Generated Clusters
#define K0X 0.2
#define K0Y 0.2
#define K1X 0.8
#define K1Y 0.2
#define K2X 0.5
#define K2Y 0.8

#define PERF_TEST // Uncomment if conducting performance test (removes prints)

using namespace std;

// Generate random clustered data
// Data is in format { x0, y0, x1, y1, x2, y2... }
void generateClusteredData(float *data) {
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

#if !defined(PERF_TEST)
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
#endif // #if !defined(PERF_TEST)

int main() {
    // Generate random data
    srand(static_cast<unsigned>(time(nullptr)));
    vector<float> data(N * D); // Data is in format { x0, y0, x1, y1, x2, y2... }
    generateClusteredData(data.data());
#if !defined(PERF_TEST)
    writeDataToCSV(data, "kmeans_data.csv");
#endif // #if !defined(PERF_TEST)
    // Run KMeans
    KMeans_CUDA model(data.data(), N, D, K); // Update here to change model

#if !defined(PERF_TEST)
    model.printCentroids();
#endif // #if !defined(PERF_TEST)
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
#if !defined(PERF_TEST)
        printf ("Epoch %i\n", epoch);
#endif // #if !defined(PERF_TEST)

        model.one_epoch();

#if !defined(PERF_TEST)
        model.printCentroids();
#endif // #if !defined(PERF_TEST)
    }
}