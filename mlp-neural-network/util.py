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


def normalization(data_set, zero_one=False):
    """
    :param data_set: numpy array
    :param zero_one: set range from zero to one if true
    :return: numpy array data with normalization
    """
    dim_x, dim_y = data_set.shape

    for y in range(dim_y):
        feature = data_set[:, y]
        max = feature.max()
        min = feature.min()
        for x in range(dim_x):
            data = data_set[x][y]
            if zero_one:
                data_set[x][y] = (data - min)/(max - min)
            else:
                data_set[x][y] = (((data - min) / (max - min)) * 0.8) + 0.1

    return data_set


def separate_data_by_class(data_set):
    """
    :param data_set: (data, labels)
    :return: set(separated_data:dict = {'label': ([x],[y])}
    """
    separated_data = {}
    total_class = 0
    data, labels = data_set
    for d, label in zip(data, labels):
        if separated_data.get(label) is None:
            separated_data[label] = ([d[0]], [d[1]])
            total_class += 1
        else:
            separated_data[label][0].append(d[0])
            separated_data[label][1].append(d[1])

    return separated_data, total_class


def visualize(data_set, data_diff=None):
    """
    :param data_set: (data, labels), want to visualize
    :param data_diff: compared data (optional) 
    :return: None
    """
    data_separated, total_class = separate_data_by_class(data_set)

    if data_diff is not None:
        data_diff, total_class_diff = separate_data_by_class(data_diff)

    # Building Color based on max class
    x = np.arange(total_class)
    ys = [i + x + (i * x) ** 2 for i in range(total_class)]
    COLORS = cm.rainbow(np.linspace(0, 1, len(ys)))

    data_visual = []
    for item in data_separated.values():
        data_visual.append(item)

    if data_diff is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

        i = 1
        for d, color in zip(data_visual, COLORS):
            x, y = d
            ax.scatter(x, y, alpha=0.8, c=color, s=30, label=i)
            i += 1

        # Shrink 10% figure from top
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        plt.title('Data set visualization')
        plt.legend(title="Class Legend", loc='upper center',ncol=int(total_class/3)+1, bbox_to_anchor=(0.5,-0.05))
        plt.show()

    else:
        f, axarr = plt.subplots(2)

        i = 1
        for d1, color in zip(data_visual, COLORS):
            x1, y1 = d1
            d2 = data_diff.get(i)
            if d2 is not None:
                x2, y2 = d2
            else:
                x2, y2 = [],[]
            axarr[0].scatter(x1, y1, alpha=0.8, c=color, s=30, label=i)
            axarr[1].scatter(x2, y2, alpha=0.8, c=COLORS[i-1], s=30, label=i)
            i += 1

        axarr[0].set_title('Real Data')
        # Shrink 10% figure from top
        box = axarr[0].get_position()
        axarr[0].set_position([box.x0, box.y0 + box.height * 0.25,
                         box.width, box.height * 0.9])

        axarr[1].set_title('Prediction Data')
        axarr[1].legend(title="Class Legend", loc='upper center',ncol=int(total_class/3)+1, bbox_to_anchor=(0.5,-0.125))
        # Shrink 10% figure from top
        box = axarr[1].get_position()
        axarr[1].set_position([box.x0, box.y0 + box.height * 0.275,
                         box.width, box.height * 0.9])

        plt.show()


def decision_boundary(data_set, classifier):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    h = 0.01
    X, labels = data_set
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    output_data = classifier.evaluate(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(output_data).reshape(xx.shape)
    ax.contourf(xx,yy,Z,cm="Paited")

    # Replot the scatter data
    data_separated, total_class = separate_data_by_class(data_set)
    data_visual = []
    for item in data_separated.values():
        data_visual.append(item)

    # Building Color based on max class
    # print(total_class)
    x = np.arange(total_class)
    ys = [i + x + (i * x) ** 2 for i in range(total_class)]
    COLORS = cm.Paired(np.linspace(0, 1, len(ys)))

    i = 1
    for d, color in zip(data_visual, COLORS):
        x, y = d
        ax.scatter(x, y, alpha=0.8, color=color, s=20, label=i)
        i += 1

    # Limit X, Y Axis
    ax.axis((0,1,0,1))

    # Shrink 10% figure from top
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    plt.title('Data set visualization')
    plt.legend(title="Class Legend", loc='upper center', ncol=int(total_class / 3) + 1, bbox_to_anchor=(0.5, -0.05))
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

            print("\nPrecision Class {} = {}".format(i + 1, precision))
            print("Recall Class {} = {}".format(i + 1, recall))
            print("F1 Score Class {} = {}".format(i + 1, f1_score))

            precision_list.append(precision)
            recall_list.append(recall)

        # F1 Score Average Class
        avg_precision = sum(precision_list) / len(precision_list)
        avg_recall = sum(recall_list) / len(recall_list)

        f1_average_score = (2 * avg_precision * avg_recall / (avg_precision + avg_recall))
        print("\nF1 Score Average (Micro)= {}".format(f1_average_score))

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
        print("\nF1 Score Average (Macro) = {}".format(f1_macro))

def view_training_graphic():
    mse_visual = np.load("training_data/mse_visual.npy").tolist()
    accuracy_visual = np.load("training_data/accuracy_visual.npy").tolist()
    eppoch = np.load("training_data/epoch.npy").tolist()
    plt.plot(range(0, eppoch), mse_visual, color='red', label="mse_error")
    plt.plot(range(0, eppoch), accuracy_visual, color='blue', label="accuracy")
    # plt.axis((0,1,0,1))
    plt.ylim((0, 1))
    plt.legend()
    plt.show()
