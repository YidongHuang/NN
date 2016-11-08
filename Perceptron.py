import random
class Perceptron:
    def __init__(self, num_next_layer_units):
        self.entropy = random()
        self.weight_list = num_next_layer_units * [None]
        for weight in self.weight_list:
            weight = random()