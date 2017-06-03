import random

import numpy as np

import k_means_clustering
import util

# Parameter
data_set_location = "datasets/Compound.csv"
total_class = 6
K = total_class

if __name__ == "__main__":

    # Load dataset
    data, labels = util.load_data_set(data_set_location, label_separated=True)
    data_set = (np.array(data), np.array(labels))

    # Visualization with no color
    util.visualize(data_set)

    # initialization cluster
    cluster = k_means_clustering.KMeansCluster(K, data_set[0])

    # initialization cluster by random point on each class
    data_separated, total_class = util.separate_data_by_class(data_set)
    class_centroid = []
    for label, data in data_separated.items():
        max_index = len(data[0])
        random_centroid = random.randint(0,max_index-1)
        class_centroid.append([data[0][random_centroid], data[1][random_centroid]])
    cluster.centroids = np.array(class_centroid)

    # clusterization process
    sse = None
    i = 1
    while True:
        new_see = cluster.clustering()
        print("Iteration {} - SSE = {}".format(i,new_see))
        if new_see == sse:
            break
        else:
            sse = new_see
        i += 1

    output_cluster = cluster.data, cluster.cluster
    util.visualize(output_cluster, centroids=cluster.centroids)