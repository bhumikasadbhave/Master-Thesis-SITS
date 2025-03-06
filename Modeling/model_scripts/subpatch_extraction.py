import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import torch
import matplotlib.pyplot as plt
import numpy as np


def non_overlapping_sliding_window(image_data, field_numbers, patch_size=5):
    """
    Apply non-overlapping sliding window to extract sub-patches, filter out zero-only sub-patches,
    and pad sub-patches with the avg(pixels) if they contain any zeros. Track the field numbers.
    """
    patches = []
    patch_coordinates = []
    batch_size, time, channels, height, width = image_data.shape
    
    # Loop over the fields (batch)
    for b in range(batch_size):  
        field_number = field_numbers[b]  

        # Extract patches across all channels and time steps in one go
        for i in range(0, height - patch_size + 1, patch_size): 
            for j in range(0, width - patch_size + 1, patch_size):  

                patch = image_data[b, :, :, i:i + patch_size, j:j + patch_size]

                if not isinstance(patch, torch.Tensor):
                    patch = torch.tensor(patch)

                if torch.any(patch > 0):  # Ignore all-zero patches
                    patch1 = patch.clone() 

                    # Calculate channel-wise mean for non-zero values
                    for t in range(time):  
                        for c in range(channels):  
                            channel_patch = patch1[t, c]            # Extract specific channel and time frame
                            if torch.any(channel_patch > 0):  
                                avg_val = torch.mean(channel_patch[channel_patch > 0])
                                channel_patch[channel_patch == 0] = avg_val  
                    
                    patches.append(patch1)
                    patch_coordinates.append((field_number, i, j))  
                        
    return torch.stack(patches), patch_coordinates



def non_overlapping_sliding_window_non_temporal(image_data, field_numbers, patch_size=5):
    """
    Apply non-overlapping sliding window to extract sub-patches, filter out zero-only sub-patches,
    and pad sub-patches with the avg(pixels) if they contain any zeros. Track the field numbers.
    
    Works for non-temporal data.
    """
    patches = []
    patch_coordinates = []
    batch_size, channels, height, width = image_data.shape  # No time dimension

    # Loop over the fields (batch)
    for b in range(batch_size):  
        field_number = field_numbers[b]  

        # Extract patches across all channels in one go
        for i in range(0, height - patch_size + 1, patch_size): 
            for j in range(0, width - patch_size + 1, patch_size):  
                
                patch = image_data[b, :, i:i + patch_size, j:j + patch_size]  # No time dimension

                if not isinstance(patch, torch.Tensor):
                    patch = torch.tensor(patch)

                if torch.any(patch > 0):  # Ignore all-zero patches
                    patch1 = patch.clone() 

                    # Calculate channel-wise mean for non-zero values
                    for c in range(channels):  
                        channel_patch = patch1[c]  # Extract specific channel
                        if torch.any(channel_patch > 0):  
                            avg_val = torch.mean(channel_patch[channel_patch > 0])
                            channel_patch[channel_patch == 0] = avg_val  
                    
                    patches.append(patch1)
                    patch_coordinates.append((field_number, i, j))  
                        
    return torch.stack(patches), patch_coordinates



def save_train_predictions_to_excel(train_field_labels, file_path):
    data = [{"Field Number": field_number, "Predicted Label": label}
            for field_number, label in train_field_labels.items()]
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)


def get_last_timestep_patches(patches):
    last_timestep_patches = patches[:, -1, :, :, :]  # Shape becomes (N, C, H, W)
    return last_timestep_patches


