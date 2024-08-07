// This is currently in C++ not CUDA for future reference

#ifndef __KMEANS_CU__
#define __KMEANS_CU__

#include <stdio.h>
#include <random>
#include <stdexcept>

using namespace std;


struct Centroid {
    float x;
    float y;
};

class KMeans_CUDA {
    public:
        
        KMeans_CUDA(
            float *data,
            int n, 
            int d,
            int k
        ) {
            this->h_data = data;
            this->n = n;
            this->d = d;
            this->k = k;

            // FIXME -- d_data

            // FIXME d classes

            // FIXME gpu centroids

            // FIXME gpu classes

            if (d != 2) {
                throw invalid_argument("Invalid d");
            }
            if (k != 3) {
                throw invalid_argument("Invalid k");
            }
            centroids = { getRandCentroid(), getRandCentroid(), getRandCentroid() };

            h_classes = new int[n];

            can_update = false;
        }

        ~KMeans_CUDA() {
            delete[] h_classes;
            // FIXME GPU
        }

        // Classifies each element
        void classify() {
            if (can_update) {
                return; // Classifications already up to date
            }

            for (int i = 0; i < n; i++) {
                h_classes[i] = classify(i);
            }

            can_update = true;
        }

        // Updates centroids
        bool update() {
            if (!can_update) {
                return false; // Need to classify before can update
            }

            // Reset
            for (Centroid& c : centroids) {
                c.x = 0;
                c.y = 0;
            }

            // Sum
            vector<size_t> lengths(k);
            for (int i = 0; i < n; i++) {
                centroids.at(h_classes[i]).x += h_data[d*i];
                centroids.at(h_classes[i]).y += h_data[d*i+1];
                lengths[h_classes[i]] += 1;
            }

            // Average
            for (int c = 0; c < k; c++) {
                if (lengths[c] == 0) {
                    centroids.at(c).x = c / k;
                    centroids.at(c).y = c / k;
                    continue;
                }
                centroids.at(c).x = centroids.at(c).x / lengths[c];
                centroids.at(c).y = centroids.at(c).y / lengths[c];
            }


            can_update = false;
            return true;
        }

        void printCentroids() {
            for (Centroid& c : centroids) {
                printf ("x: %f\ty: %f\n", c.x, c.y);
            }
        }


    private:
        float *h_data; // Data is in format { x0, y0, x1, y1, x2, y2... }. Memory stored in .cpp not class
        float *d_data; // Pointer to data on GPU

        int *h_classes;
        int *d_classes;

        vector<Centroid> centroids; // Centroids
        
        int n; // Number of data elements (i. e. { x0, y0, x1, y1, x2, y2} would be 3)
        int d; // Number of dimensions
        int k; // Number of clusters

        bool can_update;

        Centroid getRandCentroid() {
            return Centroid {
                static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                static_cast<float>(rand()) / static_cast<float>(RAND_MAX)
            };
        }

        int classify(const int i) {
            ///printf ("data x: %f, y: %f ", h_data[i*d], h_data[i*d+1]);
            int minClass = 0;
            float minDist = distance(centroids.at(0), h_data[i*d], h_data[i*d+1]);
            ///printf ("distances %f", minDist);
            for (int c = 1; c < centroids.size(); c++) {
                float dist = distance(centroids.at(c), h_data[i*d], h_data[i*d+1]);
                ///printf (" %f", dist);
                if (dist < minDist) {
                    minDist = dist;
                    minClass = c;
                }
            }
            ///printf ("\tclassification: %i\n", minClass);
            return minClass;
        }

        float distance(const Centroid &c, const float x, const float y) {
            return sqrt(
                pow((c.x - x), 2) + pow( (c.y - y), 2)
            );
        }
};

#endif // __KMEANS_CU__