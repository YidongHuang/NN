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
    mean_std = get_mean_std(train_file_meta, train_file_data)
    train_data = np.matrix(multi_sample_parser(train_file_data, train_file_meta, dict, mean_std[0], mean_std[1]))
    test_data = np.matrix(multi_sample_parser(test_file_data, train_file_meta, dict, mean_std[0], mean_std[1]))
    n_net = nn.neuro_net(learning_rate, num_hidden_units, num_epochs, train_data, test_data)
    n_net.train_nn()


def get_mean_std(meta, data):
    mean = []
    std = []
    for i in range(len(data[0])):
        if meta.types()[i] == 'numeric':
            column = [elem[i] for elem in data]
            mean.append(np.mean(column))
            std.append(np.std(column))
        else:
            mean.append(None)
            std.append(None)
    return (mean, std)


def multi_sample_parser(samples, meta, dict, mean, std):
    new_data = []
    for i in range(len(samples)):
        new_data.append(single_sample_parser(samples[i], meta, dict, mean, std))
    return new_data

def single_sample_parser(data, meta, dict, mean, std):
    parsed_data = []
    for i in range(len(meta.types())):
        if meta.types()[i] == "numeric":
            parsed_data.append((data[i] - mean[i])/std[i])
        else:
            feature_name = meta.names()[i]
            feature_range = dict[feature_name]
            hot_key = len(feature_range) * [0]
            hot_key[feature_range[data[i]]] = 1
            parsed_data.extend(hot_key)
    return parsed_data



if __name__  ==  "__main__":
    main(sys.argv[1:])