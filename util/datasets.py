# Python libraries
from abc import ABC, abstractmethod
from os import path

# External modules
import numpy as np
import pandas as pd
import cv2

# Typing
from typing import Tuple


class Dataset(ABC):

    def __init__(self, data_path: str):
        self.data_path = data_path

    
    @abstractmethod
    def next_batch_available(self) -> bool:
        pass

    
    @abstractmethod
    def get_next_batch(self, num: int) -> Tuple[np.array, np.array]:
        pass


    @abstractmethod
    def get_val(self, amount: float = 0.2) -> Tuple[np.array, np.array]:
        pass



class FirstAugmentedDataset(Dataset):

    seed = 69

    def __init__(self):
        data_path = path.abspath(path.join('data', 'generated-data'))
        super().__init__(data_path)

        data = pd.read_csv(path.join(self.data_path, 'train_aug.csv'), index_col = 0)
        self.data = data


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

        return val_X, val_y

    

def main() -> None:
    dataset = FirstAugmentedDataset()
    dataset.get_val()

    while dataset.next_batch_available(1000):
        batch_X, batch_y = dataset.get_next_batch(1000)
        print(batch_X.shape, batch_y.shape)

if __name__ == '__main__':
    main()