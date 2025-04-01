import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


######## --- Helper Functions for Modelling --- ########

def compute_avg_field_size_list(data_list):
    """
    Computes the average height and width of sugar beet fields using the field ID mask
    Input Temporal Patches - each patch has shape (T, C, H, W)
    """
    field_sizes = []

    for patch in data_list:  
        last_timestep = patch[-1]     # Last temporal instance
        mask = last_timestep[-1]      # Last channel = mask, shape

        mask = (mask > 0).astype(np.uint8)

        # Find connected components (fields)
        num_labels, label_map = cv2.connectedComponents(mask)

        for label_id in range(1, num_labels):  
            field_pixels = np.argwhere(label_map == label_id)
            if field_pixels.size > 0:
                min_h, min_w = field_pixels.min(axis=0)
                max_h, max_w = field_pixels.max(axis=0)
                height = max_h - min_h + 1
                width = max_w - min_w + 1
                field_sizes.append((height, width))

    if field_sizes:
        avg_height = np.mean([h for h, w in field_sizes])
        avg_width = np.mean([w for h, w in field_sizes])
    else:
        avg_height, avg_width = 0, 0  
    return avg_height, avg_width


def resize_with_padding(images, target_height=224, target_width=224):
    """ Resize a 64x64x3 image to 224x224x3 by adding zero-padding.
    This function is to resize our patches for feeding into pre-trained models.
    """
    padded_images = []
    for image in images:
        original_height, original_width, channels = image.shape
        pad_height = target_height - original_height
        pad_width = target_width - original_width
        assert pad_height >= 0 and pad_width >= 0, "Target size must be greater than or equal to the original size"
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_image = np.pad(image, 
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                            mode='constant', 
                            constant_values=0)
        padded_images.append(padded_image)
    return padded_images


# def resize_images(images, target_size=(224, 224)):
#     """ Preprocess Sentinel-2 images for CNN input.
#     """
#     new_images = []
#     for image in images:
#         image = image / np.max(image)  # Scale to [0, 1]
#         image = cv2.resize(image, target_size)
#         new_images.append(image)
#     return new_images
