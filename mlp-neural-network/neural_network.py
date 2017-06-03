import numpy as np


class MLPNeuralNetwork(object):
    
    def __init__(self, hidden_layer, output_layer, feature_dim, learning_rate):

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate

        # Matrix W1 (Hidden Layer), dimension = 2 x 8
        self.W1 = 2 * np.random.random((self.feature_dim, self.hidden_layer)) - 1
        # Matrix W2 (Output Layer), dimension = 8 x 31
        self.W2 = 2 * np.random.random((self.hidden_layer, self.output_layer)) - 1

        # Matrix B (Bias)
        self.B1 = 2 * np.random.random((1, self.hidden_layer)) - 1
        self.B2 = 2 * np.random.random((1, self.output_layer)) - 1


    def __activation_function(self, x: np.ndarray, dx_dy=False):
        if dx_dy is True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def __back_propagation(self, data, label=None, total_label=None, **kwargs):

        # Forward Hidden & Output Layer Calculation
        data = np.array([data])
        V1 = np.dot(data, self.W1) + self.B1
        A1 = self.__activation_function(V1)  # dimension sesuai neuron hidden layer 1x8
        V2 = np.dot(A1, self.W2) + self.B2
        A2 = self.__activation_function(V2)  # dimension sesuai jumlah neuron output 1x31

        if kwargs.get("test") is True:
            return A2

        # output_neuron_label, for counting error
        output_neuron = np.zeros((1, total_label))
        output_neuron[0][label - 1] = 1
        E = output_neuron - A2  # dimension 1 x 31 (neuron output)
        mse = np.sqrt(np.sum(np.power(E, 2))) ** 2

        # Back Propagation Start Here
        D2 = self.__activation_function(A2, dx_dy=True) * E  # dimension 1x31
        D1 = self.__activation_function(A1, dx_dy=True) * np.dot(D2, self.W2.T)  # dimension 1x8
        dW2 = self.learning_rate * A1.T.dot(D2)
        dW1 = self.learning_rate * data.T.dot(D1)
        # Bias
        dB1 = self.learning_rate * D1
        dB2 = self.learning_rate * D2

        # update weight
        self.W2 += dW2
        self.W1 += dW1
        self.B1 += dB1
        self.B2 += dB2

        if kwargs.get("mse_training") is True:
            return mse
        else:
            return 0

    def training_phase(self, data_set, labels, MSE=False):
        total_label = len(set(labels))  # 31 Checked
        total_data = len(labels)
        mse = 0
        for data, label in zip(data_set, labels):
            mse += self.__back_propagation(data, label, total_label, mse_training=MSE)
        return mse / total_data

    def predict(self, data, debug=False):
        A2 = self.__back_propagation(data, test=True)
        if debug:
            print(A2)
        # Predict the data
        return np.where(A2 == A2.max())[1][0]+1

    def evaluate(self, data_no_label):
        labels = []
        for data in data_no_label:
            prediction = self.predict(data)
            labels.append(prediction)
        return labels

    def test_accuracy(self, data_set, conf_matrix=False):
        valid, error = 0, 0
        data, labels = data_set

        # make confusion matrix based on total label (label x label)
        count_label = len(set(labels))
        confusion_matrix = [[0 for i in range(count_label)] for i in range(count_label)]

        for data, label in zip(data, labels):
            predict = self.predict(data)
            if label == predict:
                valid += 1
            else:
                error += 1

            confusion_matrix[label - 1][predict - 1] += 1

        accuracy = round(valid / (valid + error), 4)
        if conf_matrix is True:
            return confusion_matrix
        else:
            return accuracy
