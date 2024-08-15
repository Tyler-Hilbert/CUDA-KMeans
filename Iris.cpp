// Main class for running Iris K-Means with ArrayFire

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "KMeans_ArrayFire.h"

using namespace std;

#define D 4 // Number of dimensions (not including class)
#define K 3 // Number of clusters


// Only used on CPU for parsing CSV
struct IrisData {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    char species;
};

// Turn CSV into vector
// Will be turned into structure of arrays before passed to GPU
vector<IrisData> readCSV(const string& filename) {
    vector<IrisData> data;
    ifstream file(filename);
    string line;
    // Skip the header line
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string temp;
        IrisData iris;

        getline(ss, temp, ',');
        iris.sepal_length = stof(temp);

        getline(ss, temp, ',');
        iris.sepal_width = stof(temp);

        getline(ss, temp, ',');
        iris.petal_length = stof(temp);

        getline(ss, temp, ',');
        iris.petal_width = stof(temp);

        getline(ss, temp, ',');
        if (temp == "Iris-setosa") {
            iris.species = 0;
        } else if (temp == "Iris-versicolor") {
            iris.species = 1;
        } else {
            iris.species = 2;
        }

        data.push_back(iris);
    }

    return data;
}

int main() {
    string filename = "iris.csv";
    vector<IrisData> iris_data = readCSV(filename);

    const int N = iris_data.size();
    char  *h_species = new char[N]; // Used for debugging
    float *h_data =    new float[N*D];

    // Convert vector to structure of arrays
    // Index of each field
    const int sli = 0;
    const int swi = N;
    const int pli = N*2;
    const int pwi = N*3;
    int i = 0;
    for (IrisData &iris : iris_data) {
        h_data[i+sli] = iris.sepal_length;
        h_data[i+swi] = iris.sepal_width;
        h_data[i+pli] = iris.petal_length;
        h_data[i+pwi] = iris.petal_width;
        h_species[i++] = iris.species;
    }

    // Run KMeans on GPU
    try {
        KMeans_ArrayFire model (h_data, N, D, K);

        //model.print_centroids();

        for (int i = 0; i < 10; i++) {
            model.one_epoch();
            model.print_predictions();
            //model.print_centroids();
        }
    } catch (af::exception& e) { 
        fprintf(stderr, "%s\n", e.what()); throw; 
    }


    // Delete heap memory
    delete [] h_species;
    delete[] h_data;
    
    return 0;
}