import numpy as np
import random as random
import math as math


class neuro_net:
    def __init__(self, learning_rate, num_hidden_units, num_epochs, train_data, test_data):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_data = train_data
        np.random.shuffle(self.train_data)
        self.input_layer = None
        self.hidden_layer_error = None
        self.hidden_layer = None
        self.output_error = -1
        self.output = -1
        self.num_hidden_units = num_hidden_units
        self.train_size = train_data.shape
        self.test_data = test_data
        if num_hidden_units > 0:
            self.hidden_layer = np.random.uniform(-0.01, 0.01, num_hidden_units + 1).tolist()
            self.input_layer = np.random.rand(num_hidden_units, self.train_size[1] - 1) * 0.02 - 0.01
            # self.hidden_layer = np.random.uniform(1, 1, num_hidden_units + 1).tolist()
            # self.input_layer = np.random.rand(num_hidden_units, self.train_size[1] - 1) * 0.02 - 0.01
        else:
            self.input_layer =  np.random.uniform(-0.01, 0.01, (self.train_size[1] -1 )).tolist()
            # self.input_layer =  np.random.uniform(1, 1, (self.train_size[1] -1 )).tolist()
        self.train_features = self.train_data[:, 0:self.train_size[1] - 2]
        self.train_labels = self.train_data[:, -1]
        self.test_features = self.test_data[:, 0:self.train_size[1] - 2]
        self.test_labels = self.test_data[:, -1]
        # self.print_matrix()

    def print_matrix(self):
        print self.train_features
        print self.train_labels
        print self.input_layer
        print self.hidden_layer

    def train_nn(self):
        if self.num_hidden_units > 0:
            self.train_nn_with_hidden_unit()
        else:
            self.train_nn_no_hidden_unit()


    def train_nn_with_hidden_unit(self):
        tests_over_epoch = []
        for i in range(self.num_epochs):
            for j in range(self.train_size[0]):
                # print "****************************************"
                train_label = self.train_labels[j].item(0)
                train_sample = self.train_features[j,:].tolist()[0] + [1]
                hidden_net = np.matmul(self.input_layer, train_sample)
                # print "input_layer is {}".format(self.input_layer)
                # print "train sample is {}".format(train_sample)
                # print "hidden net is {}".format(hidden_net)
                # print "label is {}".format(train_label)
                hidden_unit_input = self.get_sigmoid_list(hidden_net.tolist() + [1])
                # print "hidden unit input is {}".format(hidden_unit_input)
                net = np.dot(self.hidden_layer, hidden_unit_input)
                # print "net is {}".format(net)
                output = self.get_sigmoid(net)
                # print "output is {}".format(output)
                delta_output = train_label - output
                # print "delta output is {}".format(delta_output)
                # print "step is {}".format(np.multiply(self.learning_rate * delta_output, hidden_unit_input))
                # print "hidden layer is {}".format(self.hidden_layer)
                hidden_layer_derivative = [hidden_unit_input[m] * (1-hidden_unit_input[m]) for m in range(len(hidden_unit_input))]
                delta_hidden_unit = [hidden_layer_derivative[m] * np.multiply(delta_output,self.hidden_layer)[m] for m in range(len(hidden_layer_derivative))]
                # print "hidden layer derivative is {}".format(hidden_layer_derivative)
                # print "delta_hidden_unit is {}".format(delta_hidden_unit)
                self.hidden_layer = np.add(self.hidden_layer, np.multiply(self.learning_rate * delta_output, hidden_unit_input))
                # print "updated hidden_layer is {}".format(self.hidden_layer)
                for k in range(self.num_hidden_units):
                    # print "input layer {} is {}".format(k, self.input_layer[k,:])
                    self.input_layer[k,:] = np.add(self.input_layer[k,:], np.multiply(self.learning_rate * delta_hidden_unit[k], train_sample))
                    # print "input layer changed to {}".format(self.input_layer[k,:])
            test = self.test_with_hidden_unit()
            print "epoch time is {}, error is {}, correctly predicted {}, incorrectly predicted {}".format(i, test[2],test[0],test[1])
            tests_over_epoch.append(test)
        return 0


    def train_nn_no_hidden_unit(self):
        tests_over_epoch = []
        for i in range(self.num_epochs):
            for j in range(self.train_size[0]):
                train_label = self.train_labels[j].item(0)
                train_sample = [1] + self.train_features[j,:].tolist()[0]
                # print "input layer is {}, train_sample is {}, train_label is {}".format(self.input_layer, train_sample, train_label)
                net = np.dot(self.input_layer, train_sample)
                output = self.get_sigmoid(net)
                delta = train_label - output
                # print "net is {}, output is {}, delta is {}".format(net, output, delta)
                self.input_layer = np.add(self.input_layer, np.multiply(self.learning_rate * delta ,train_sample))
                # print "step is {}, self.input_layer is {}".format(np.multiply(self.learning_rate * delta ,train_sample), self.input_layer)
            test = self.test_no_hidden_unit()
            print "epoch time is {}, error is {}, correctly predicted {}, incorrectly predicted {}".format(i, test[2], test[0], test[1])
            tests_over_epoch.append(test)
        return 0

    def get_sigmoid(self, net):
        return 1/(1 + math.exp( - net))

    def get_sigmoid_list(self, list_net):
        sigmoid_list = []
        for net in list_net:
            sig = self.get_sigmoid(net)
            sigmoid_list.append(sig)
        return sigmoid_list

    def test_no_hidden_unit(self):
        num_correct = 0
        num_incorrect = 0
        error = 0
        for i in range(len(self.test_features)):
            label = self.test_labels[i].item(0)
            test_sample = [1] + self.test_features[i].tolist()[0]
            net = np.dot(self.input_layer, test_sample)
            output = self.get_sigmoid(net)
            prediction = round(output)
            # print prediction
            error += -label * np.log(output) - (1-label) * np.log(1 - output)
            if prediction == label:
                num_correct += 1
            else:
                num_incorrect +=1
        return (num_correct, num_incorrect, error)

    def test_with_hidden_unit(self):
        num_correct = 0
        num_incorrect = 0
        error = 0
        for j in range(len(self.test_labels)):
            label = self.test_labels[j].item(0)
            test_sample = self.test_features[j,:].tolist()[0] + [1]
            hidden_net = np.matmul(self.input_layer, test_sample)
            # print 'input layer is {}'.format(self.input_layer)
            # print 'train_sampe is {}'.format(train_sample)
            # print 'hidden net is {}'.format(hidden_net)
            hidden_unit_input = self.get_sigmoid_list(hidden_net.tolist() + [1])
            # print hidden_unit_input
            net = np.dot(self.hidden_layer, hidden_unit_input)
            # print 'hidden layer is {}'.format(self.hidden_layer)
            # print 'hidden layer input is {}'.format(hidden_unit_input)
            # print 'net is {}'.format(net)
            output = self.get_sigmoid(net)
            prediction = round(output)
            error += -label * np.log(output) - (1 - label) * np.log(1 - output)
            if prediction == label:
                num_correct += 1
            else:
                num_incorrect += 1
        return (num_correct, num_incorrect, error)