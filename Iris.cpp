 // Main class for running Iris K-Means 

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

//#include "KMeans_ArrayFire.h"
#include "KMeans_CUDA.h"

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
    if (D != 4) {
        throw invalid_argument("Invalid D");
    }

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

// Converts vector to a structure of arrays
void vectorToSoA(vector<IrisData> &iris_data, int N, char *h_species, float *h_data){
    if (D != 4) {
        throw invalid_argument("Invalid D");
    }

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
}

// Convert vector to array of structures
void vectorToAoS(vector<IrisData> &iris_data, int N, char *h_species, float *h_data){
    if (D != 4) {
        throw invalid_argument("Invalid D");
    }
    
    int i = 0;
    int j = 0;
    for (IrisData &iris : iris_data) {
        h_data[i++] = iris.sepal_length;
        h_data[i++] = iris.sepal_width;
        h_data[i++] = iris.petal_length;
        h_data[i++] = iris.petal_width;
        h_species[j++] = iris.species;
    }
}

int main() {
    string filename = "iris.csv";
    vector<IrisData> iris_data = readCSV(filename);

    const int N = iris_data.size();
    char  *h_species = new char[N]; // Used for debugging
    float *h_data =    new float[N*D];

    //vectorToSoA(iris_data, N, h_species, h_data); // ArrayFire
    vectorToAoS(iris_data, N, h_species, h_data);

    // ArrayFire
    /*
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
    */

   // My CUDA K-Means Implementation
    KMeans_CUDA model (h_data, N, D, K);
    model.print_centroids();
    for (int i = 0; i < 10; i++) {
        printf ("Epoch %i\n", i);
        model.one_epoch();
        model.print_predictions();
        model.print_centroids();
    }

    
    // Delete heap memory
    delete [] h_species;
    delete[] h_data;
    
    return 0;
}