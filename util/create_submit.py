# Python libraries
import csv

# External modules
import torch
import pandas as pd


def create_submit(
    pred_y: torch.Tensor, 
    names: pd.DataFrame,
    save_path: str    
) -> None:
    pred_y = pred_y.numpy()
    pred_y = pd.DataFrame(pred_y, columns = [
        'healthy', 
        'multiple_diseases', 
        'rust', 
        'scab'
    ])
    pred_y['image_id'] = names.image_id

    pred_y.to_csv(save_path, quoting = csv.QUOTE_MINIMAL)
