import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FieldDataset(Dataset):
    def __init__(self, inputs, field_numbers):
        # if isinstance(inputs, np.ndarray):
        if len(inputs.shape) == 4:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W) -> to account for non-temporal data
        elif len(inputs.shape) == 5:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)   # (N, T, C, H, W) -> (N, C, T, H, W)
        #inputs = torch.tensor(inputs, dtype=torch.float32)
        self.inputs = inputs
        self.field_numbers = field_numbers       

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.field_numbers[idx]
    
def create_data_loader(inputs, field_numbers, batch_size=32, shuffle=True):
    dataset = FieldDataset(inputs, field_numbers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader


class FieldDatasetMAE(Dataset):
    def __init__(self, inputs, field_numbers, timestamps):
        # if isinstance(inputs, np.ndarray):
        # if len(inputs.shape) == 4:
        #     inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)      # (N, H, W, C) -> (N, C, H, W) -> to account for non-temporal data
        # elif len(inputs.shape) == 5:
        #     inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)   # (N, T, C, H, W) -> (N, C, T, H, W)
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        self.inputs = inputs
        self.field_numbers = field_numbers 
        if timestamps is not None:
            self.timestamps = timestamps         

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.field_numbers[idx], self.timestamps[idx]
    
def create_data_loader_mae(inputs, field_numbers, timestamps, batch_size=64, shuffle=True):
    dataset = FieldDatasetMAE(inputs, field_numbers, timestamps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

### --- Augmentation code --- ###

class FieldDatasetMAEAug(Dataset):
    def __init__(self, inputs, field_numbers, timestamps, augmented_flags, augmentations=None):
        """
        Dataset class for handling augmented field images.
        inputs: Tensor of shape (N, C, T, H, W)
        field_numbers: List of field identifiers
        timestamps: List of timestamps for images
        augmented_flags: List of booleans indicating if the image was augmented
        augmentations: Torchvision transform to apply augmentations
        """
        # Ensure correct tensor formatting
        if len(inputs.shape) == 4:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        elif len(inputs.shape) == 5:
            inputs = torch.tensor(inputs, dtype=torch.float32).permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)

        self.inputs = inputs
        self.field_numbers = field_numbers
        self.timestamps = timestamps
        self.augmented_flags = augmented_flags
        self.augmentations = augmentations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns a tuple (image, field_number, timestamp, augmented_flag)
        """
        image = self.inputs[idx]  # Shape: (C, T, H, W)
        field_number = self.field_numbers[idx]
        timestamp = self.timestamps[idx]
        augmented = self.augmented_flags[idx]

        # Apply augmentations if specified
        if self.augmentations:
            transformed_images = []
            for t in range(image.shape[1]):  # Loop over time dimension
                img_t = transforms.ToPILImage()(image[:, t, :, :])  # Convert to PIL
                img_t = self.augmentations(img_t)  # Apply augmentation
                transformed_images.append(transforms.ToTensor()(img_t))  # Convert back to tensor

            # Stack along the time dimension
            image = torch.stack(transformed_images, dim=1)  # Shape: (C, T, H, W)

        return image, field_number, timestamp, augmented


def create_augmented_data_loader(dataloader, augmentations, batch_size=64, shuffle=True, keep_original=True):
    """
    Creates a DataLoader with augmented images.
    keep_original: If True, keeps original images along with augmented ones.
    """
    augmented_inputs = []
    augmented_field_numbers = []
    augmented_timestamps = []
    augmented_flags = []

    for images, field_numbers, timestamps in dataloader:
        # print(images.shape)
        for i in range(images.shape[0]):  # Iterate over batch
            img = images[i]  # (C, T, H, W)
            # print(img.shape)

            # Append original image if keeping originals
            if keep_original:
                augmented_inputs.append(img)
                augmented_field_numbers.append(field_numbers[i])
                augmented_timestamps.append(timestamps[i])
                augmented_flags.append(False)  # Not augmented

            for aug in augmentations:
                transformed_images = []
                for t in range(img.shape[1]):  # Loop over T (time steps)
                    # print(img.shape)
                    img_t = transforms.ToPILImage()(img[:, t, :, :])  # Convert to PIL
                    img_t = aug(img_t)  # Apply specific augmentation
                    transformed_images.append(transforms.ToTensor()(img_t))  # Convert back to tensor
                img_aug = torch.stack(transformed_images, dim=1)  # Shape: (C, T, H, W)

                augmented_inputs.append(img_aug)
                augmented_field_numbers.append(field_numbers[i])
                augmented_timestamps.append(timestamps[i])
                augmented_flags.append(True)  

    augmented_inputs = torch.stack(augmented_inputs, dim=0)    
    dataset = FieldDatasetMAEAug(augmented_inputs, augmented_field_numbers, augmented_timestamps, augmented_flags, augmentations=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def get_augmentation_transforms():
    """ Returns a composition of augmentation transforms.
    """
    return [
        transforms.RandomHorizontalFlip(p=1.0),  # flip
        transforms.RandomRotation(90),          # rotate by 90 degrees
    ]


def remove_augmented_images(dataloader):
    """
    Removes augmented images from the dataset to revert to original data.
    """
    original_inputs = []
    original_field_numbers = []
    original_timestamps = []

    for images, field_numbers, timestamps, augmented_flags in dataloader:
        for i in range(len(augmented_flags)):
            if not augmented_flags[i]:  # Keep only non-augmented samples
                original_inputs.append(images[i])
                original_field_numbers.append(field_numbers[i])
                original_timestamps.append(timestamps[i])

    original_inputs = torch.stack(original_inputs, dim=0) if original_inputs else torch.tensor([])
    return original_inputs, original_field_numbers, original_timestamps
