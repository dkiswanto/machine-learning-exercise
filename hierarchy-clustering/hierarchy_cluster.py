import math
import itertools

import numpy as np


def calculate_distance(x1, x2):
    """
    :param x1: numpy array (vector)
    :param x2: numpy array (vector)
    :return: distance between vector
    """
    if hasattr(x1, '__iter__'):
        x1 = np.array(x1)
    else:
        x1 = np.array([x1])

    if hasattr(x2, '__iter__'):
        x2 = np.array(x2)
    else:
        x2 = np.array([x2])

    data = (x1 - x2) ** 2
    return math.sqrt(sum(data))


def single_link(cluster1, cluster2):
    min_distance = None
    for point_a in cluster1:
        for point_b in cluster2:
            dist = calculate_distance(point_a, point_b)
            if min_distance is None:
                min_distance = dist
            else:
                if dist < min_distance:
                    min_distance = dist
    return min_distance


def complete_link(cluster1, cluster2):
    max_distance = None
    for point_a in cluster1:
        for point_b in cluster2:
            dist = calculate_distance(point_a, point_b)
            if max_distance is None:
                max_distance = dist
            else:
                if dist > max_distance:
                    max_distance = dist
    return max_distance


def group_average(cluster1, cluster2):
    total_distance = 0.0
    for point_a in cluster1:
        for point_b in cluster2:
            dist = calculate_distance(point_a, point_b)
            total_distance += dist
    return total_distance / (len(cluster1) * len(cluster2))


def centroid_based(cluster1, cluster2):
    centroid_1 = np.array(cluster1).mean(axis=0)
    centroid_2 = np.array(cluster2).mean(axis=0)
    # print((centroid_1,centroid_2))
    return calculate_distance(centroid_1, centroid_2)


def agglomerative_clustering(data_set, type):
    current_cluster = data_set
    output_cluster = []
    i = 1
    while len(current_cluster) > 1:
        shortest_distance = None
        shortest_object = None
        # print(current_cluster,'\n')
        best_cluster = None
        for data in itertools.combinations(current_cluster, 2):
            # try:
            cluster1, cluster2 = data
            if np.array(cluster1).shape == (2,):
                cluster1 = [cluster1]
            if np.array(cluster2).shape == (2,):
                cluster2 = [cluster2]

            if type == 'single_link' or type == 1:
                dist = single_link(cluster1,cluster2)

            elif type == 'complete_link' or type == 2:
                dist = complete_link(cluster1,cluster2)

            elif type == 'group_average' or type == 3:
                dist = group_average(cluster1,cluster2)

            elif type == 'centroid_based' or type == 4:
                dist = centroid_based(cluster1,cluster2)
            else:
                print("Please select similarity measure method type")
                return None

            if shortest_distance is None:
                shortest_distance = dist
                shortest_object = data
                best_cluster = cluster1, cluster2
            else:
                if dist < shortest_distance:
                    shortest_distance = dist
                    shortest_object = data
                    best_cluster = cluster1, cluster2

        current_cluster.remove(shortest_object[0])
        current_cluster.remove(shortest_object[1])
        temp = []
        for point in best_cluster[0]: temp.append(point)
        for point in best_cluster[1]: temp.append(point)
        current_cluster.append(temp)
        # print(current_cluster)
        # output_cluster.remove(shortest_object[0])
        # output_cluster.remove(shortest_object[1])
        # output_cluster.append(shortest_object)
        print("Cluster-{}\n{}\nDistance: {}\n".format(i, list(shortest_object), round(shortest_distance,2)))
        i += 1
        # break
    return output_cluster