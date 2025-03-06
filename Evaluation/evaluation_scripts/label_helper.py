import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np


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

    # Aggregate predictions for each field based on the threshold
    field_labels = {}
    
    for field_number, predictions in field_dict.items():
        diseased_patch_count = np.sum(np.array(predictions) == 1)
        field_labels[field_number] = 1 if diseased_patch_count >= (threshold * len(predictions)) else 0

    return field_labels


def evaluate_test_labels(test_field_labels, ground_truth_csv_path):
    """
    Compare predicted patch-level evaluation labels with the ground truth from a CSV file.
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

    for field_number, predicted_label in updated_test_field_labels.items():
        if field_number in ground_truth:
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(1 if true_label == "yes" else 0)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, report, cm


################################ functions for AE #######################################

def assign_field_labels_ae(subpatch_coordinates, subpatch_predictions, threshold=0.1):
    """
    Assign patch/field-level labels based on sub-patch predictions
    Returns: field_labels: Dictionary {field_number: field_label}
    """
    field_dict = {}
    for field_number, prediction in zip(subpatch_coordinates, subpatch_predictions):
        if field_number not in field_dict:
            field_dict[field_number] = []
        field_dict[field_number].append(prediction)

    field_labels = {}    
    for field_number, predictions in field_dict.items():
        diseased_subpatch_count = np.sum(np.array(predictions) == 1)
        field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

    return field_labels


def evaluate_test_labels_ae(test_field_labels, ground_truth_csv_path):
    """
    Compare predicted field labels with ground truth loaded from a CSV file.
    Extract last two numbers as (x, y) coordinates and map them separately.
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
        
        x_y_coords[field_number, (int(float(x)), int(float(y)))] = label

    y_pred = []
    y_true = []

    for field_number, predicted_label in updated_test_field_labels.items():
        if field_number in ground_truth:
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(1 if true_label == "yes" else 0)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, report, cm, x_y_coords


################################ functions for DCEC #######################################

def assign_field_labels_clustering(subpatch_coordinates, subpatch_predictions, threshold=0.1):
    """
    Assign patch/field-level labels based on sub-patch clustering predictions.
    Returns: field_labels: Dictionary {field_number: field_label}
    """
    field_dict = {}
    
    # Assign each sub-patch to the predicted cluster (highest probability in q)
    for field_number, prediction in zip(subpatch_coordinates, subpatch_predictions):
        # Find the index of the highest cluster assignment probability
        cluster_label = prediction  # Using the cluster with max probability
        if field_number not in field_dict:
            field_dict[field_number] = []
        field_dict[field_number].append(cluster_label)

    field_labels = {}    
    for field_number, predictions in field_dict.items():
        # Count the number of sub-patches that belong to the diseased cluster (cluster 1, for example)
        diseased_subpatch_count = np.sum(np.array(predictions) == 1)  # Assuming cluster 1 is the diseased cluster
        
        # Assign the label based on the threshold
        field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

    return field_labels
