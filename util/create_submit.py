# Python libraries
import csv

# External modules
import torch
import pandas as pd
import numpy as np


def create_submit(
    pred_y: np.array, 
    names: np.array,
    save_path: str    
) -> None:
    pred_y = pd.DataFrame(pred_y)
    pred_y.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
    pred_y['image_id'] = names
    pred_y = pred_y[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]
    pred_y.to_csv(save_path, index = False, quoting = csv.QUOTE_MINIMAL)
