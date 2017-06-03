from sys import argv

import hierarchy_cluster
import util

data_set_location = "dataset/Hierarchical_2.csv"

if __name__ == "__main__":

    # Load dataset
    data = util.load_data_set(data_set_location)

    # Visualization with no color
    util.visualize(data)

    # Argument Disimmiliarity method type clustering from CLI
    # ex : python3 main 1
    # 1 for single link
    # 2 for complete link
    # 3 for group average
    # 4 for centroid based
    try:
        type = int(argv[1])
    except:
        type = 1    # default 1 for no argument
    hierarchy_cluster.agglomerative_clustering(data, type=type)
