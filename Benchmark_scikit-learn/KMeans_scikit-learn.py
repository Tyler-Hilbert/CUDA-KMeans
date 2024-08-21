# KMeans in scikit-learn for comparison.

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

iris = pd.read_csv("../TestData/iris.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values
kmeans = KMeans(
    n_clusters = 3,
    init = 'k-means++',
    max_iter = 10,
    n_init = 1,
    random_state = 0
)
y_kmeans = kmeans.fit_predict(x)
print (y_kmeans)