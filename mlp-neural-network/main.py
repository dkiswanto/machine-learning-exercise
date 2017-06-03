import numpy as np
import os.path

import params
import util
from neural_network import MLPNeuralNetwork

# Load data set & Normalization
data, labels = util.load_data_set(params.data_set, label_separated=True)
data, labels = util.normalization(np.array(data)), np.array(labels)
data_set = (data, labels)

# Visualize data set first
util.visualize(data_set)

# Make classifier
classifier = MLPNeuralNetwork(params.hidden_layer, params.output_layer,
                              params.feature_dim, params.learning_rate)

# Load weight data if exist
if os.path.exists("training_data/W1.npy") and os.path.exists("training_data/W2.npy") \
        and os.path.exists("training_data/B1.npy") and os.path.exists("training_data/B2.npy"):
    classifier.W1 = np.load("training_data/W1.npy")
    classifier.W2 = np.load("training_data/W2.npy")
    classifier.B1 = np.load("training_data/B1.npy")
    classifier.B2 = np.load("training_data/B2.npy")

# Classify data
predict_labels = classifier.evaluate(data)
predict_data_set = (data, predict_labels)

# visualize prediction data
util.visualize(data_set, predict_data_set)

# visualize boundary
util.decision_boundary(data_set, classifier)

conf_matrix = classifier.test_accuracy(data_set, conf_matrix=True)
util.performance_calculation(conf_matrix, mode="f1_micro_average")
accuracy = util.performance_calculation(conf_matrix, mode="accuracy")

print("Accuracy = {}".format(accuracy))

# visualize graphic training
util.view_training_graphic()

print(classifier.W1)
print(classifier.W2)
print(classifier.B1)
print(classifier.B2)