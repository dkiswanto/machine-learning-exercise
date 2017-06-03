import numpy as np
import os.path

import params
import util
from neural_network import MLPNeuralNetwork
from matplotlib import pyplot as plt

# Load data set & Normalization
data, labels = util.load_data_set(params.data_set, label_separated=True)
data, labels = util.normalization(np.array(data)), np.array(labels)
data_set = (data, labels)

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

# Training classifier
minimum_error = 0.2
error = 100.0
acc = 0.0
if os.path.exists("training_data/accuracy_visual.npy") and os.path.exists("training_data/mse_visual.npy") \
    and os.path.exists("training_data/epoch.npy"):
    mse_visual = np.load("training_data/mse_visual.npy").tolist()
    accuracy_visual = np.load("training_data/accuracy_visual.npy").tolist()
    eppoch = np.load("training_data/epoch.npy").tolist()
else:
    mse_visual = []
    accuracy_visual = []
    eppoch = 0

# Training Phase
while error > 0.5:
    try:
        error = classifier.training_phase(data, labels, MSE=True)
        acc = classifier.test_accuracy(data_set)
        print("Epoch {} - MSE={}, ACC={}".format(eppoch, error, acc))
        mse_visual.append(error)
        accuracy_visual.append(acc)
        eppoch += 1

    # When stop save weight & bias data.
    except KeyboardInterrupt:
        np.save("training_data/W1.npy", classifier.W1)
        np.save("training_data/W2.npy", classifier.W2)
        np.save("training_data/B1.npy", classifier.B1)
        np.save("training_data/B2.npy", classifier.B2)
        np.save("training_data/epoch.npy", np.array(eppoch))
        np.save("training_data/mse_visual.npy", np.array(mse_visual))
        np.save("training_data/accuracy_visual.npy", np.array(accuracy_visual))

        plt.plot(range(0,eppoch), mse_visual, color='red', label="mse_error")
        plt.plot(range(0,eppoch), accuracy_visual, color='blue', label="accuracy")
        # plt.axis((0,1,0,1))
        plt.ylim((0,1))
        plt.legend()
        plt.show()

np.save("training_data/W1.npy", classifier.W1)
np.save("training_data/W2.npy", classifier.W2)
np.save("training_data/B1.npy", classifier.B1)
np.save("training_data/B2.npy", classifier.B2)
np.save("training_data/epoch.npy", np.array(eppoch))
np.save("training_data/mse_visual.npy", np.array(mse_visual))
np.save("training_data/accuracy_visual.npy", np.array(accuracy_visual))

print("Training Done, Saved")