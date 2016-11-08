from scipy.io.arff import loadarff
import sys
import numpy as np
import neuro_net as nn

def main(argv):
    learning_rate = float(argv[0])
    num_hidden_units = int(argv[1])
    num_epochs = int(argv[2])
    train_file_data, train_file_meta = loadarff(argv[3])
    test_file_data, test_file_meta = loadarff(argv[4])

    dict = {}
    for feature_name in train_file_meta.names():
        if train_file_meta[feature_name][0] == 'numeric':
            continue
        else:
            feature_range = {}
            for i in range(len(train_file_meta[feature_name][1])):
                feature_range[train_file_meta[feature_name][1][i]] = i
            dict[feature_name] = feature_range
    train_data = multi_sample_parser(train_file_data, train_file_meta, dict)
    test_data = multi_sample_parser(test_file_data, train_file_meta, dict)
    train_data = np.matrix(standardizer(train_data))
    test_data = np.matrix(standardizer(test_data))

    n_net = nn.neuro_net(learning_rate, num_hidden_units, num_epochs, train_data)
    n_net.test(test_data)
    n_net.print_matrix()


def standardizer(data):
    num_col = len(data[0])
    for i in range(num_col - 1):
        column = [elem[i] for elem in data]
        mean = np.mean(column)
        std = np.std(column)
        if std == 0.0:
            continue
        for sample in data:
            sample[i] = (float(sample[i] - mean))/std
    return data


def multi_sample_parser(samples, meta, dict):
    new_data = []
    for i in range(len(samples)):
        new_data.append(single_sample_parser(samples[i], meta, dict))
    return new_data

def single_sample_parser(data, meta, dict):
    parsed_data = []
    for i in range(len(meta.types())):
        if meta.types()[i] == "numeric":
            parsed_data.append(data[i])
        else:
            feature_name = meta.names()[i]
            feature_range = dict[feature_name]
            hot_key = len(feature_range) * [0]
            hot_key[feature_range[data[i]]] = 1
            parsed_data.extend(hot_key)
    return parsed_data



if __name__  ==  "__main__":
    main(sys.argv[1:])