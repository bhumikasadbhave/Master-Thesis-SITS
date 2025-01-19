import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FieldDataset(Dataset):
    def __init__(self, inputs, field_numbers):
        # if isinstance(inputs, np.ndarray):
        if len(inputs.shape) == 4:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        elif len(inputs.shape) == 5:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)
        #inputs = torch.tensor(inputs, dtype=torch.float32)
        
        self.inputs = inputs
        self.field_numbers = field_numbers          # field_numbers are strings

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.field_numbers[idx]



def create_data_loader(inputs, field_numbers, batch_size=32, shuffle=True):
    dataset = FieldDataset(inputs, field_numbers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



# class FieldDatasetPatches(Dataset):
#     def __init__(self, inputs, field_numbers):
#         # if isinstance(inputs, np.ndarray):
#         if len(inputs.shape) == 4:
#             inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#         elif len(inputs.shape) == 5:
#             inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)
#         #inputs = torch.tensor(inputs, dtype=torch.float32)
        
#         self.inputs = inputs
#         self.field_numbers = field_numbers          # field_numbers are strings

#     def __len__(self):
#         return len(self.inputs)
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.field_numbers[idx]


# def create_data_loader_patches(inputs, field_numbers, batch_size=32, shuffle=True):
#     dataset = FieldDataset(inputs, field_numbers)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader