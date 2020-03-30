# Python libraries
import os
import csv

# External modules
import pandas as pd


def get_128px_train_data():
    imgs_path = os.path.abspath(os.path.join('data', 'generated-data', 'train-128.csv'))
    train_X = pd.read_csv(imgs_path, quoting = csv.QUOTE_MINIMAL, index_col = 0)

    labels_path = os.path.join('data', 'raw-data', 'train.csv')
    train_y = pd.read_csv(labels_path, quoting = csv.QUOTE_MINIMAL)

    return train_X, train_y


def get_128px_test_data():
    imgs_path = os.path.join('data', 'generated-data', 'test-128.csv')
    test_X = pd.read_csv(imgs_path, quoting = csv.QUOTE_MINIMAL, index_col = 0)

    labels_path = os.path.join('data', 'raw-data', 'test.csv')
    test_y = pd.read_csv(labels_path, quoting = csv.QUOTE_MINIMAL)

    return test_X, test_y
