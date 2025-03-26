import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FieldDataset(Dataset):
    def __init__(self, inputs, field_numbers, timestamps):
        # if isinstance(inputs, np.ndarray):
        if len(inputs.shape) == 4:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W) -> to account for non-temporal data
        elif len(inputs.shape) == 5:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)   # (N, T, C, H, W) -> (N, C, T, H, W)
        #inputs = torch.tensor(inputs, dtype=torch.float32)
        
        self.inputs = inputs
        self.field_numbers = field_numbers 
        if timestamps is not None:
            self.timestamps = timestamps         

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if self.timestamps is not None:
            return self.inputs[idx], self.field_numbers[idx], self.timestamps[idx]
        else: 
            return self.inputs[idx], self.field_numbers[idx]



def create_data_loader(inputs, field_numbers, batch_size=32, shuffle=True):
    dataset = FieldDataset(inputs, field_numbers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_data_loader_mae(inputs, field_numbers, timestamps, batch_size=64, shuffle=True):
    dataset = FieldDataset(inputs, field_numbers, timestamps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader