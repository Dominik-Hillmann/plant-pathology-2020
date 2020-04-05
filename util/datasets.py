# Python libraries
from abc import ABC, abstractmethod
from os import path
from math import floor

# External modules
import numpy as np
import pandas as pd
import cv2

# Typing
from typing import Tuple


class Dataset(ABC):
    """Base class representing all classes that part the data into validation, 
    training data and into batches."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    
    @abstractmethod
    def next_batch_available(self, n: int) -> bool:
        """Checks wether `n` observations are still available."""
        pass

    
    @abstractmethod
    def get_next_batch(self, num: int) -> Tuple[np.array, np.array]:
        """Get a batch of `n` observations."""
        pass


    @abstractmethod
    def get_val(self, amount: float = 0.2) -> Tuple[np.array, np.array]:
        """Get the validation data. By default 20%."""
        pass


    @abstractmethod
    def num_possible_batches(self, size: int) -> int:
        """Get the number of possible batches of size `size`."""
        pass



class FirstAugmentedDataset(Dataset):

    seed = 69

    def __init__(self):
        data_path = path.abspath(path.join('data', 'generated-data'))
        super().__init__(data_path)

        data = pd.read_csv(path.join(self.data_path, 'train_aug.csv'), index_col = 0)
        self.data = data
        self.val_data_subtracted = False


    def next_batch_available(self, n: int) -> bool:
        return n <= len(self.data)


    def get_next_batch(self, n: int) -> Tuple[np.array, np.array]:
        batch_data = self.data.sample(n = n, random_state = self.seed)
        self.data = self.data.drop(batch_data.index)

        batch_y = batch_data.label_index.to_numpy()

        img_names = batch_data.image_id.tolist()
        imgs = []
        for img_name in img_names:
            img_path = path.abspath(path.join(self.data_path, 'augmented-data-128px', img_name))
            imgs.append(cv2.imread(img_path))
        batch_X = np.array(imgs)

        return batch_X, batch_y


    def get_val(self, amount: float = 0.2) -> Tuple[np.array, np.array]:
        val_data = self.data.sample(frac = amount, random_state = self.seed)
        self.data = self.data.drop(val_data.index)        

        val_y = val_data.label_index.to_numpy()
        
        img_names = val_data.image_id.tolist()
        imgs = []
        for img_name in img_names:
            img_path = path.abspath(path.join(self.data_path, 'augmented-data-128px', img_name))
            imgs.append(cv2.imread(img_path))
        val_X = np.array(imgs)

        self.val_data_subtracted = True
        return val_X, val_y
    

    def num_possible_batches(self, batch_size: int) -> int:
        """Get the number of possible batches of size `batch_size`."""

        if self.val_data_subtracted:
            return floor(len(self.data) / batch_size)
        else:
            raise ValueError('First get validation data.')
