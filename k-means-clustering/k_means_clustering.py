import math
import random

import numpy as np


class KMeansCluster(object):

    @staticmethod
    # def init_centroid(min, max, data_shape):
    #     return np.random.uniform(min, max, (data_shape))
    def init_centroid(feature_dim, data_shape):
        x, y = data_shape
        min_max_1, min_max_2 = feature_dim
        generated_centroid = [
            [random.uniform(min_max_1[0],min_max_1[1]),
             random.uniform(min_max_2[0], min_max_2[1])]
            for i in range(x)
        ]
        return np.array(generated_centroid)

    @staticmethod
    def min_max_feature(data):
        """
        :param data: data point without label.
        :return: return max and min for all feature. : (max, min)
        """
        dim_x, dim_y = data.shape
        # max_all, min_all = None, None
        feature_dim = []
        for y in range(dim_y):
            feature = data[:, y]
            max = feature.max()
            min = feature.min()
            # if max_all is None:
            #     max_all = max
            #     min_all = min
            # else:
            #     if max > max_all:
            #         max_all = max
            #     if min < min_all:
            #         min_all = min
            feature_dim.append((min,max))
        # return min_all, max_all
        return feature_dim

    @staticmethod
    def calculate_distance(x1, x2):
        """
        :param x1: numpy array (vector)
        :param x2: numpy array (vector)
        :return: distance between vector
        """
        data = (x1 - x2) ** 2
        return math.sqrt(sum(data))

    def __init__(self, k, data):
        self.data = data

        # generate random centroid (c)
        feature_min_max = self.min_max_feature(data)
        self.centroids = self.init_centroid(feature_min_max, (k,2))

        # make cluster assignment of data vector (a)
        x, y = data.shape
        self.cluster = np.zeros(x, dtype=np.uint8)

    def clustering(self):

        cluster_set = {}    # Cj

        for index, point in enumerate(self.data):

            nearest = None
            for label, centroid in enumerate(self.centroids):
                # print("centroid {}".format(centroid))
                # print("point {}".format(point))

                distance = self.calculate_distance(point, centroid)
                if nearest is None:
                    nearest = distance, point, label
                else:
                    if distance < nearest[0]:
                        nearest = distance, point, label

            self.cluster[index] = nearest[2] # label nearest
            if cluster_set.get(nearest[2]) is None:
                cluster_set[nearest[2]] = [nearest[1]]
            else:
                cluster_set[nearest[2]].append(nearest[1])

        SSE = 0

        # Update Centroid
        for i, key in enumerate(cluster_set.keys()):
            data_cluster = cluster_set.get(key)
            new_centroid = sum(data_cluster) / len(data_cluster)
            self.centroids[key] = new_centroid

            # SSE Count
            for data in data_cluster:
                SSE += self.calculate_distance(new_centroid,data) ** 2

        return SSE

