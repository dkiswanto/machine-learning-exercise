import csv
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval

from matplotlib import cm


def load_data_set(filename, label_separated=False):
    data = []
    label = []
    with open(filename) as f:
        reader = csv.reader(f)
        # next(reader, None)  # skip header
        for row in reader:
            # change data type (using list comprehension)
            if label_separated is False:
                data.append([literal_eval(i) for i in row])
            else:
                data.append((float(row[0]), float(row[1])))
                label.append(int(row[2]))
    if label_separated is False:
        return data
    else:
        return data, label

def visualize(data_set, centroids=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    i = ord('A')
    for x,y in data_set:
        ax.scatter(x, y, alpha=0.8, c='grey', s=30, edgecolor='black', label=chr(i))
        ax.annotate(chr(i), xy=(x+0.2,y+0.2))
        i += 1

    plt.title('Data set visualization')
    x, y = -5, 20
    plt.axis((x,y,x,y))
    plt.xticks(np.arange(x, y, 1))
    plt.yticks(np.arange(x, y, 1))
    # plt.grid()
    plt.show()


