import numpy as np
import random as random
class neuro_net:
    def __init__(self, learning_rate, num_hidden_units, num_epochs, train_data):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_data = np.random.shuffle(train_data)
        self.input_layer = None
        self.hidden_layer_error = None
        self.hidden_layer = None
        self.output_error = -1
        self.output = -1
        if num_hidden_units > 0:
            self.hidden_layer = np.array(num_hidden_units * [random.uniform(-0.01, 0.01)])
            self.hidden_Layer_error = num_hidden_units * [-1]
        if self.hidden_layer is not None:
            self.input_layer = np.random.rand(num_hidden_units, train_data.shape[1]) * 0.02 - 0.01

    def print_matrix(self):
        print self.input_layer
        print self.hidden_layer

    def test(self, test_data):
        return 0

