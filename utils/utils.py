import torch
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split


DATA_PATH = '/Users/shared/data/MNIST/train.csv'

def get_raw_data():
	data_raw = genfromtxt(DATA_PATH, delimiter = ',', dtype = np.float32)
	data = data_raw[1:, :]

	feature_data_raw = data[:, 1:]
	label_data_raw = data[:, 0]

	return feature_data_raw, label_data_raw

def get_partitioned_data(feature_data_raw, label_data_raw):
	feature_data_raw, label_data_raw = get_raw_data()

	feature_data_raw = feature_data_raw / 255.0

	features_train_np, features_test_np, targets_train_np, targets_test_np = train_test_split(feature_data_raw, 
																								label_data_raw, 
																								test_size = 0.2, 
																								random_state = 5)

	features_train = torch.from_numpy(features_train_np)
	targets_train = torch.from_numpy(targets_train_np).type(torch.LongTensor)
	features_test = torch.from_numpy(features_test_np)
	targets_test = torch.from_numpy(targets_test_np).type(torch.LongTensor)

	return features_train, targets_train, features_test, targets_test


def get_data_loader(features_train, 
					targets_train, 
					features_test, 
					targets_test, 
					batch_size = 256):
	rs_features_train = features_train.view(-1, 1, 28, 28).float()
	rs_features_test = features_test.view(-1, 1, 28, 28).float()

	train = torch.utils.data.TensorDataset(rs_features_train, targets_train)
	test = torch.utils.data.TensorDataset(rs_features_test, targets_test)

	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
	test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

	return train_loader, test_loader

def get_misclassified(net, test_loader):
    features_misclass = []
    labels_misclass = []

    with torch.no_grad():
        net.eval()
        for features, labels in test_loader:
            outputs = net(features)
            _, preds = outputs.max(1)
            misclass = ~preds.eq(labels)
            features_misclass += features[misclass]
            labels_misclass += labels[misclass]
            
    return features_misclass, labels_misclass