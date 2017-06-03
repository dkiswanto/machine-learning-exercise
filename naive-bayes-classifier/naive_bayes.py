import math
import numpy as np


class NaiveBayes(object):

    def __init__(self, data_set):
        self.prior = self.prior_calculation(data_set)
        self.likelihood = self.likelihood_calculation(data_set)
        self.current_dataset = data_set

    def prior_calculation(self, data_set):
        LABEL_INDEX = 2
        total_data = len(data_set)
        count_label = {}

        for data in data_set:

            label = data[LABEL_INDEX]

            if count_label.get(label) is None:
                count_label[label] = 1
            else:
                count_label[label] += 1

        for key_label in count_label.keys():

            count_label[key_label] = float(count_label[key_label]) / total_data

        return count_label # as prior

    def likelihood_calculation(self, data_set):

        # using Algorithm: Continuous-valued Features
        LABEL_INDEX = 2
        collection_label = {'x1': {}, 'x2': {}}

        for data in data_set:

            label = data[LABEL_INDEX]

            if collection_label['x1'].get(label) is None:
                collection_label['x1'][label] = [data[0]]
                collection_label['x2'][label] = [data[1]]

            else:
                collection_label['x1'][label].append(data[0])
                collection_label['x2'][label].append(data[1])

        # likelihood_models = {}
        # = {"x1" : {1 : [mean, std], 2 : [mean, std]}}

        model = {'x1': {}, 'x2': {}}
        for dimen in collection_label.keys():

            for data in collection_label[dimen].items():

                label = data[0]
                np_array = np.array(data[1])
                mean = np_array.mean()
                std = np_array.std(ddof=1)
                model[dimen][label] = (mean, std)

        return model

    def gaussian_model(self, x, mean, std):
        left = (1 / (math.sqrt(2 * math.pi) * std))
        inside = -(math.pow(x-mean, 2) / (2 * math.pow(std, 2)))
        right = math.exp(inside)
        return right * left

    def classify(self, data):
        probability = {}
        for label in self.prior:
            mean_x1, std_x1 = self.likelihood['x1'][label]
            mean_x2, std_x2 = self.likelihood['x2'][label]

            x1_prob = self.gaussian_model(data[0], mean_x1, std_x1)
            x2_prob = self.gaussian_model(data[1], mean_x2, std_x2)

            posterior = math.log(x1_prob) + math.log(x2_prob) + math.log(self.prior[label])
            probability[label] = posterior

        # Implement Map Rules
        max_prob = None
        for prob in probability.items():

            if max_prob is None:
                max_prob = prob
            else:
                if max_prob[1] < prob[1]:
                    max_prob = prob

        return max_prob[0] #just return the label

    def evaluate(self, data_set, mode="with_matrix"):

        new_data = []
        count_label = 6
        confusion_matrix = [[0 for i in range(count_label)] for i in range(count_label)]
        for data in data_set:
            if not mode == "no_matrix":
                real_label = data[2]

            prediction = self.classify(data)
            new_data.append([data[0], data[1], prediction])

            if not mode == "no_matrix":
                confusion_matrix[real_label-1][prediction-1] += 1

        return new_data, confusion_matrix

