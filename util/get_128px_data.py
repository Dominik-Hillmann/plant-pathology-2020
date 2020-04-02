# Python libraries
import os
import csv

# External modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
SEED = 28

# Typing
from typing import Tuple


def one_hot_to_idx(frame: pd.DataFrame) -> pd.DataFrame:
    frame['label_index'] = 0
    frame.loc[frame.healthy == 1, 'label_index'] = 0
    frame.loc[frame.multiple_diseases == 1, 'label_index'] = 1
    frame.loc[frame.rust == 1, 'label_index'] = 2
    frame.loc[frame.scab == 1, 'label_index'] = 3

    frame = frame.drop(['healthy', 'multiple_diseases', 'rust', 'scab'], axis = 'columns')
    return frame


def get_128px_train_data(val_size: float = 0.2) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    imgs_path = os.path.abspath(os.path.join('data', 'generated-data', 'train-128.csv'))
    train_X = pd.read_csv(imgs_path, quoting = csv.QUOTE_MINIMAL, index_col = 0)

    labels_path = os.path.join('data', 'raw-data', 'train.csv')
    train_y = pd.read_csv(labels_path, quoting = csv.QUOTE_MINIMAL)

    train_X.index = train_X.image_id
    train_X = train_X.sort_index()
    train_X = train_X.drop('image_id', axis = 'columns')
    train_y.index = train_y.image_id
    train_y = train_y.sort_index()
    train_y = train_y.drop('image_id', axis = 'columns')

    train_y = one_hot_to_idx(train_y)

    train_X, val_X, train_y, val_y = train_test_split(
        train_X, 
        train_y, 
        test_size = val_size,
        random_state = SEED
    )

    print(
        'Loaded data with shapes of training and validation X and ys:',
        train_X.shape, train_y.shape, val_X.shape, val_y.shape
    )
    return train_X.values, train_y.values, val_X.values, val_y.values


def get_128px_test_data() -> Tuple[np.array, np.array]:
    imgs_path = os.path.join('data', 'generated-data', 'test-128.csv')
    test_X = pd.read_csv(imgs_path, quoting = csv.QUOTE_MINIMAL, index_col = 0)

    labels_path = os.path.join('data', 'raw-data', 'test.csv')
    test_y = pd.read_csv(labels_path, quoting = csv.QUOTE_MINIMAL)

    test_X.index = test_X.image_id
    test_X = test_X.sort_index()
    test_X = test_X.drop('image_id', axis = 'columns')
    test_y.index = test_y.image_id
    test_y = test_y.sort_index()
    test_y = test_y.drop('image_id', axis = 'columns')

    return test_X.values, test_y.values
