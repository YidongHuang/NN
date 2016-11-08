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
        self.num_hidden_units = num_hidden_units
        self.train_size = train_data.shape
        if num_hidden_units > 0:
            self.hidden_layer = np.array(num_hidden_units * [random.uniform(-0.01, 0.01)])
            self.hidden_Layer_error = num_hidden_units * [-1]
        if self.hidden_layer is not None:
            self.input_layer = np.random.rand(num_hidden_units, train_data.shape[1]) * 0.02 - 0.01

    def print_matrix(self):
        print self.input_layer
        print self.hidden_layer

    def train_nn(self):
        if self.num_hidden_units > 0:
            self.train_nn_with_hidden_unit()
        else:
            self.train_nn_no_hidden_unit()


    def train_nn_with_hidden_unit(self):
        for i in range(self.num_epochs):

        return 0

    def train_nn_no_hidden_unit(self):
        for i in range(self.num_epochs):
            for j in range(self.train_size[0]):
                train_entry = [1]
                output = np.matmul(self.input_layer, train_entry.extend(self.train_data[j, 0:self.train_size[1] - 1]))
                error = self.train_data[j: self.train_size[1]] - output
                self.input_layer = np.subtract(self.input_layer, self.learning_rate * error * self.input_layer)
        return 0

    def get_output(self, weight, val):
        return np.matmul(weight, val)

    def test(self, test_data):
        return 0

