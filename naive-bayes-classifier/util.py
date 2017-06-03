import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from ast import literal_eval

COLORS = ("red", "green", "blue", 'black', 'orange', 'white')
EDGES = ("none", "none", "none", 'none', 'none', 'grey')
GROUPS = ("Class 1", " Class 2", "Class 3", 'Class 4', 'Class 5', "Class 6")


def load_data_set(filename):
    data = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        # next(reader, None)  # skip header
        for row in reader:

            # change data type (using list comprehension)
            data.append([literal_eval(i) for i in row])

    return data


def separate_data_by_class(data_set):
    LABEL_INDEX = 2
    separated_data = {}

    for data in data_set:

        label = data[LABEL_INDEX]

        if separated_data.get(label) is None:
            separated_data[label] = ([data[0]], [data[1]])
        else:
            separated_data[label][0].append(data[0])
            separated_data[label][1].append(data[1])

    return separated_data


def visualize(data_set):
    data_set = separate_data_by_class(data_set)

    data = []
    for item in data_set.values():
        data.append(item)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    for d, color, edge, group in zip(data, COLORS, EDGES, GROUPS):
        x, y = d
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors=edge, s=30, label=group)

    plt.title('Compound Dataset Visualization')
    plt.legend(loc=3)
    plt.show()


def compare_data(data_one, data_two):
    f, axarr = plt.subplots(2, sharex=True)

    data_one = separate_data_by_class(data_one)
    data_two = separate_data_by_class(data_two)

    data_one_visual = []
    for item in data_one.values():
        data_one_visual.append(item)

    data_two_visual = []
    for item in data_two.values():
        data_two_visual.append(item)

    for d1, d2, color, edge, group in zip(data_one_visual, data_two_visual, COLORS, EDGES, GROUPS):
        x1, y1 = d1
        x2, y2 = d2
        axarr[0].scatter(x1, y1, alpha=0.8, c=color, edgecolors=edge, s=30, label=group)
        axarr[1].scatter(x2, y2, alpha=0.8, c=color, edgecolors=edge, s=30, label=group)

    axarr[0].set_title('Real Data')
    axarr[1].set_title('Prediction Data')

    # axarr[1].pcolormesh([1,2])
    # axarr[1].contour([1,2], 1, 1, [0.5], linewidths=0.75, colors='k')

    plt.show()


def decision_boundary(data_set, classifier):
    h = 1
    X = np.array(data_set)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # print np.c_[xx.ravel(), yy.ravel()]
    data_label, matrix = classifier.evaluate(np.c_[xx.ravel(), yy.ravel()], mode="no_matrix")
    data_visual = np.array(data_label)[:,2].reshape(xx.shape)
    plt.contourf(xx, yy, data_visual, cmap=plt.cm.Paired)

    data_set = separate_data_by_class(data_set)

    data = []
    for item in data_set.values():
        data.append(item)

    for d, color, edge, group in zip(data, COLORS, EDGES, GROUPS):
        x, y = d
        plt.scatter(x, y, alpha=0.8, c=color, edgecolors=edge, s=30, label=group)

    plt.axis('off')
    plt.show()


def performance_calculation(matrix, mode="accuracy"):

    matrix = np.array(matrix)
    if mode == "accuracy":
        positive = 0
        for i in range(len(matrix)):
            positive += matrix[i][i]

        sum_eval = matrix.sum()

        return round(float(positive) / sum_eval, 3)

    elif mode == "f1_micro_average":
        precision_list = []
        recall_list = []
        for i in range(len(matrix)):
            true_positive = matrix[i, i]

            precision = float(true_positive) / sum(matrix[i]) # oke
            recall = float(true_positive) / sum(matrix[:,i])
            f1_score = (2 * precision * recall) / (precision + recall)

            print "\nPrecision Class {} = {}".format(i + 1, precision)
            print "Recall Class {} = {}".format(i + 1, recall)
            print "F1 Score Class {} = {}".format(i + 1, f1_score)

            precision_list.append(precision)
            recall_list.append(recall)

        # F1 Score Average Class
        avg_precision = sum(precision_list) / len(precision_list)
        avg_recall = sum(recall_list) / len(recall_list)

        f1_average_score = (2 * avg_precision * avg_recall / (avg_precision + avg_recall))
        print "\nF1 Score Average (Micro)= {}".format(f1_average_score)

    elif mode == "f1_macro_average":
        precision_list = []
        recall_list = []
        for i in range(len(matrix)):
            true_positive = matrix[i, i]

            precision = float(true_positive) / sum(matrix[i]) # oke
            recall = float(true_positive) / sum(matrix[:,i])

            precision_list.append(precision)
            recall_list.append(recall)

        # F1 Score Average Class
        avg_precision = sum(precision_list) / len(precision_list)
        avg_recall = sum(recall_list) / len(recall_list)

        f1_macro = avg_precision + avg_recall / 2
        print "\nF1 Score Average (Macro) = {}".format(f1_macro)

