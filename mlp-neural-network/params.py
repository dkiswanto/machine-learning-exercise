import math
import util

data_set = "datasets/D31.csv"
label = util.load_data_set(data_set,label_separated=True)[1]

# Initialization data MLP Neural Network
feature_dim = 2
output_layer = len(set(label))
# hidden_layer = round(math.sqrt(feature_dim * output_layer)) # Takes so long when training data.
hidden_layer = len(set(label)) # more faster in training phase
learning_rate = 0.01
# learning_rate = 0.001    # 87%

