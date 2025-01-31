import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import cv2
from torchvision import transforms
from PIL import Image, ImageDraw
import os


def non_overlapping_sliding_window(image_data, field_numbers, patch_size=5):
    """
    Apply non-overlapping sliding window to extract patches, filter out zero-only patches,
    and pad patches with the max(pixels) if they contain zeros. Track the field numbers.
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
                    patch1 = patch.clone()  # Create a copy to modify
                    # avg_val = torch.mean(patch1[patch1 > 0]) if torch.any(patch1 > 0) else 0
                    # patch1[patch1 == 0] = avg_val

                    # Calculate channel-wise mean for non-zero values
                    for t in range(time):  
                        for c in range(channels):  
                            channel_patch = patch1[t, c]            # Extract specific channel and time frame
                            if torch.any(channel_patch > 0):  
                                avg_val = torch.mean(channel_patch[channel_patch > 0])
                                channel_patch[channel_patch == 0] = avg_val  
                    

                    patches.append(patch1)
                    patch_coordinates.append((field_number, i, j))  # Track field and spatial coordinates
                        
    return patches, patch_coordinates


def train_kmeans_patches(train_patches, n_clusters, random_state):
    flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()  
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(flattened_patches)
    return kmeans


def assign_field_labels(patch_coordinates, patch_predictions, threshold=0.1):
    """
    Assign field-level labels based on patch predictions.
    Returns: field_labels: Dictionary {field_number: field_label}.
    """
    field_dict = {}
    for (field_number, _, _), prediction in zip(patch_coordinates, patch_predictions):
        if field_number not in field_dict:
            field_dict[field_number] = []
        field_dict[field_number].append(prediction)

    # Aggregate predictions for each field
    field_labels = {}
    # for field_number, predictions in field_dict.items():
    #     field_labels[field_number] = int(np.any(np.array(predictions) == 1)) 
    
    for field_number, predictions in field_dict.items():
        diseased_patch_count = np.sum(np.array(predictions) == 1)
        field_labels[field_number] = 1 if diseased_patch_count >= (threshold * len(predictions)) else 0

    return field_labels


def evaluate_test_labels(test_field_labels, ground_truth_csv_path):
    """
    Compare predicted field labels with ground truth loaded from a CSV file.
    """
    df = pd.read_csv(ground_truth_csv_path, sep=';')
    ground_truth = {
        str(row["Number"]): row["Disease"].strip().lower()  
        for _, row in df.iterrows()
    }
    updated_test_field_labels = {}
    for field_number, label in test_field_labels.items():
        if '_' in field_number:
            split_field_numbers = field_number.split('_')
            for split_field in split_field_numbers:
                updated_test_field_labels[str(int(float(split_field)))] = label
        else:
            updated_test_field_labels[str(int(float(field_number)))] = label

    y_pred = []
    y_true = []
    # for field_number, true_label in ground_truth.items():
    #     if field_number in updated_test_field_labels:
    #         y_pred.append(updated_test_field_labels[field_number])
    #         y_true.append(1 if true_label == "yes" else 0)
    # print(updated_test_field_labels,ground_truth)
    for field_number, predicted_label in updated_test_field_labels.items():
        if field_number in ground_truth:
            # print('something')
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(1 if true_label == "yes" else 0)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report


def save_train_predictions_to_excel(train_field_labels, file_path):
    data = [{"Field Number": field_number, "Predicted Label": label}
            for field_number, label in train_field_labels.items()]
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)


def visualize_single_patch_temporal_rgb(patch, patch_coordinates, patch_size=5, num_of_timestamps=7):

    field_number, i, j = patch_coordinates  
    fig, axes = plt.subplots(1, num_of_timestamps, figsize=(20, 5))
    fig.suptitle(f"Field {field_number} - Patch ({i}, {j})", fontsize=16)
    for t in range(num_of_timestamps):
        rgb_image = np.stack([patch[t, 0], patch[t, 1], patch[t, 2]], axis=-1)  
        min_val = np.min(rgb_image)
        max_val = np.max(rgb_image)
        if max_val > min_val:
            rgb_image = (rgb_image - min_val) / (max_val - min_val)  
        rgb_image = np.clip(rgb_image, 0, 1)  
        axes[t].imshow(rgb_image, cmap='viridis') 
        axes[t].set_title(f"Timestamp {t + 1}", fontsize=10)
        axes[t].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.show()


def visualize_patches(image_data, field_numbers_test, patch_coordinates, field_index, patch_size=5):

    num_of_img_train, t, channels, height, width = image_data.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    img = image_data[field_index, -1, 0].cpu().numpy()
    img_fn = field_numbers_test[field_index]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Field {field_index}, Last Time Step")
    ax.axis("off")
    for field_number, i, j in patch_coordinates:
        if field_number == img_fn:
            rect = patches.Rectangle((j, i), patch_size, patch_size, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


def get_last_timestep_patches(patches):
    last_timestep_patches = patches[:, -1, :, :, :]  # Shape becomes (N, C, H, W)
    return last_timestep_patches


################################ functions for AE #######################################

def assign_field_labels_ae(patch_coordinates, patch_predictions, threshold=0.1):
    """
    Assign field-level labels based on patch predictions.
    Returns: field_labels: Dictionary {field_number: field_label}.
    """
    field_dict = {}
    for field_number, prediction in zip(patch_coordinates, patch_predictions):
        if field_number not in field_dict:
            field_dict[field_number] = []
        field_dict[field_number].append(prediction)

    field_labels = {}    
    for field_number, predictions in field_dict.items():
        diseased_patch_count = np.sum(np.array(predictions) == 1)
        field_labels[field_number] = 1 if diseased_patch_count >= (threshold * len(predictions)) else 0

    return field_labels


def evaluate_test_labels_ae(test_field_labels, ground_truth_csv_path):
    """
    Compare predicted field labels with ground truth loaded from a CSV file.
    Extracts last two numbers as (x, y) coordinates and maps them separately.
    """
    df = pd.read_csv(ground_truth_csv_path, sep=';')
    ground_truth = {
        str(row["Number"]): row["Disease"].strip().lower()  
        for _, row in df.iterrows()
    }

    updated_test_field_labels = {}
    x_y_coords = {}  

    for field_number, label in test_field_labels.items():

        split_field_numbers = field_number.split('_')
        x, y = split_field_numbers[-2], split_field_numbers[-1]
        field_ids = split_field_numbers[:-2]  
        for field_id in field_ids:
            updated_test_field_labels[str(int(float(field_id)))] = label
        
        x_y_coords[(int(float(x)), int(float(y)))] = label

    y_pred = []
    y_true = []

    for field_number, predicted_label in updated_test_field_labels.items():
        if field_number in ground_truth:
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(1 if true_label == "yes" else 0)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return accuracy, report, x_y_coords



def draw_diseased_patches(dataloader, x_y_coords, save_path="output/"):
    os.makedirs(save_path, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        images, field_ids = batch

        for img_idx, (img, field_id) in enumerate(zip(images, field_ids)):
            field_id = str(field_id)

            img = img[img_idx, :3, -1, :, :]  # Use last time step

            if img.shape[0] >= 3:
                img_np = img[:3].permute(1, 2, 0).numpy() * 255
            else:
                img_np = img.mean(dim=0).numpy() * 255

            img_np = img_np.astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_pil)

            split_field_numbers = field_id.split('_')
            x, y = int(float(split_field_numbers[-2])), int(float(split_field_numbers[-1]))

            if (x, y) in x_y_coords and x_y_coords[(x, y)] == 1:
                rect_size = 5
                top_left = (x - rect_size // 2, y - rect_size // 2)
                bottom_right = (x + rect_size // 2, y + rect_size // 2)
                draw.rectangle([top_left, bottom_right], outline="red", width=2)

            img_pil.save(f"{save_path}/batch{batch_idx}_img{img_idx}_field_{field_id}.png")
            print(f"Saved: batch{batch_idx}_img{img_idx}_field_{field_id}.png")


