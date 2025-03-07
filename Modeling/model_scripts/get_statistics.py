import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

def get_accuracy(field_numbers, labels, gt_path):

    df = pd.read_csv(gt_path, sep=';')
    gt_fn = df['Number'].tolist()
    gt_label = df['Disease'].tolist()
    gt_mapping = {int(float(gt_fn[i])): gt_label[i].strip().lower() for i in range(len(gt_fn))}

    field_labels = {}

    for i in range(len(field_numbers)):

        number = field_numbers[i]
        if '_' in number:
            all_numbers = number.split('_')
            for n in all_numbers:
                field_labels[int(float(n))]=labels[i]
        else:
            field_labels[int(float(number))] = labels[i]

    # Calculate Accuracy
    correct = 0
    total = len(field_labels)
    for field_number, predicted_label in field_labels.items():

        ground_truth = gt_mapping.get(field_number, None)
        if (predicted_label == 1 and ground_truth == 'yes') or (predicted_label == 0 and ground_truth == 'no'):
            correct += 1

    # Create and return aligned preds and ground truths
    gt_aligned = []
    pred_aligned = []
    for field_number, predicted_label in field_labels.items():
        if field_number in gt_mapping:
            gt_aligned.append(1 if gt_mapping[field_number] == 'yes' else 0)
            pred_aligned.append(predicted_label)
    
    accuracy = correct / total if total > 0 else 0.0
    report = classification_report(gt_aligned, pred_aligned)
    cm = confusion_matrix(gt_aligned, pred_aligned)

    return accuracy, report, cm, pred_aligned, gt_aligned

def compute_histograms(stack, bins=10):
    """Compute histograms for a temporal stack of images.
    Args: stack: Array of shape (7, 64, 64, 3), temporal stack of images.
          bins: Number of bins for the histogram.
    Returns:
        hist_features: 1D array of concatenated histograms.
    """
    temporal_histograms = []
    for img in stack: 
        histograms = []
        for channel in range(3):  
            hist, _ = np.histogram(img[..., channel], bins=bins, range=(0, 255), density=True)
            histograms.append(hist)
        temporal_histograms.append(np.concatenate(histograms))  # Histogram per channel

    # Mean across temporal dimension
    hist_features = np.mean(temporal_histograms, axis=0)  
    return hist_features

def resize_with_padding(images, target_height=224, target_width=224):
    """ Resize a 64x64x3 image to 224x224x3 by adding zero-padding.
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

def resize_images(images, target_size=(224, 224)):
    """ Preprocess Sentinel-2 images for CNN input.
    """
    new_images = []
    for image in images:
        image = image / np.max(image)  # Scale to [0, 1]
        image = cv2.resize(image, target_size)
        new_images.append(image)
    return new_images


def normalize_temporal_image(image):
    """
    Normalize each channel of a temporal image to the range [0, 1].
    Useless data with pixel value 0 will be ignored during normalization.

    Parameters: image (np.ndarray): A 3D array representing a temporal image (Height, Width, Channels).
    Returns: np.ndarray: Normalized image of the same shape.
    """

    normalized_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        channel_data = image[:, :, channel]

        # Ignore pixels with value 0 (mask them)
        valid_pixels = channel_data[channel_data > 0]
        if valid_pixels.size > 0:
            channel_min = valid_pixels.min()
            channel_max = valid_pixels.max()

            if channel_max > channel_min:
                normalized_channel = (channel_data - channel_min) / (channel_max - channel_min)
                # Mask useless data (keep 0 for useless pixels)
                normalized_channel[channel_data == 0] = 0
                normalized_image[:, :, channel] = normalized_channel
            else:
                # If channel_min == channel_max, keep the channel as 0
                normalized_image[:, :, channel] = 0
        else:
            # If all pixels are 0, leave the channel as 0
            normalized_image[:, :, channel] = 0

    return normalized_image


### Helper Function for getting average height and width of sugar-beet fields (for the Manuscript) ###

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