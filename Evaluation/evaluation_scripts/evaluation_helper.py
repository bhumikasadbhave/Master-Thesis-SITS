import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, recall_score, f1_score, precision_score
import matplotlib.patches as patches
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import torch
import matplotlib.pyplot as plt
import config
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score, homogeneity_completeness_v_measure


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
    Compare predicted subpatch-level evaluation labels with the ground truth from a CSV file.
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


def evaluate_clustering_metrics_old(test_field_labels, ground_truth_csv_path):
    """
    Evaluate clustering accuracy (ACC), precision, recall, and F1-score per class.
    Uses the Hungarian algorithm to find the best mapping between predicted clusters and ground-truth labels.
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
            true_label = 1 if ground_truth[field_number] == "yes" else 0
            y_pred.append(predicted_label)
            y_true.append(true_label)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Build confusion matrix
    unique_clusters = np.unique(y_pred)
    unique_labels = np.unique(y_true)
    num_clusters = len(unique_clusters)
    num_labels = len(unique_labels)
    
    cost_matrix = np.zeros((num_clusters, num_labels))

    for i, cluster in enumerate(unique_clusters):
        for j, label in enumerate(unique_labels):
            cost_matrix[i, j] = -np.sum((y_pred == cluster) & (y_true == label))

    # Hungarian algorithm -> Best assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Best cluster-label mapping
    mapping = {unique_clusters[row]: unique_labels[col] for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([mapping[pred] for pred in y_pred])

    acc = np.mean(mapped_preds == y_true)
    cm = confusion_matrix(y_true, mapped_preds, labels=unique_labels)

    # precision_per_class = np.diag(cm) / np.sum(cm, axis=0, where=(np.sum(cm, axis=0) != 0))
    # recall_per_class = np.diag(cm) / np.sum(cm, axis=1, where=(np.sum(cm, axis=1) != 0))
    # f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    # fmi = fowlkes_mallows_score(y_true, mapped_preds)

    precision = precision_score(y_true, mapped_preds)
    f1 = f1_score(y_true, mapped_preds)
    recall = recall_score(y_true, mapped_preds)
    f2_score = fbeta_score(y_true, mapped_preds, beta=2)

    return acc, precision, recall, f1, f2_score


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


################################ New functions for assigning labels without Hungarian #######################################

def evaluate_clustering_metrics(subpatch_coordinates, subpatch_predictions, ground_truth_csv_path, threshold=0.5):
    """
    Evaluate clustering accuracy (ACC), precision, recall, and F1-score per class.
    Compares both assumptions (1=disease and 0=disease) and selects the best cluster mapping.
    """
    # Ground truths from CSV
    df = pd.read_csv(ground_truth_csv_path, sep=';')
    ground_truth = {
        str(row["Number"]): 1 if row["Disease"].strip().lower() == "yes" else 0
        for _, row in df.iterrows()
    }
    
    # Prepare the patch level labels: output of this step = {'field#': [0,1,0,0,..], ..} 
    # ie field number and the subpatch-predictions for that field
    field_with_subpatch_labels = {}
    for field_number, label in zip(subpatch_coordinates, subpatch_predictions):
        split_field_numbers = field_number.split('_')
        for i in range(len(split_field_numbers)-2):
            field_number = str(int(float(split_field_numbers[i])))
            if field_number not in field_with_subpatch_labels:
                field_with_subpatch_labels[field_number] = []
            field_with_subpatch_labels[field_number].append(label)


    ## Scenario 1: cluster 1=disease
    # Aggregate subpatch level predictions to patch level labels assuming 1=disease
    # output of this step = {'field#': 0, 'field#': 1, .. }
    field_labels = {}    
    for field_number, predictions in field_with_subpatch_labels.items():
        diseased_subpatch_count = np.sum(np.array(predictions) == 1)        #total number of '1' sub-patch labels
        field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

    # Arrange predicted and true labels in lists
    y_pred = []
    y_true = []
    for field_number, predicted_label in field_labels.items():
        if field_number in ground_truth:
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(true_label)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    acc_1 = np.mean(y_pred == y_true)
    recall_1 = recall_score(y_true, y_pred)
    precision_1 = precision_score(y_true, y_pred)
    f1_1 = f1_score(y_true, y_pred)
    f2_1 = fbeta_score(y_true, y_pred, beta=2)
        

    ## Scenario 2: cluster 0=disease
    # Aggregate subpatch level predictions to patch level labels assuming 1=disease
    # output of this step = {'field#': 0, 'field#': 1, .. }
    field_labels = {}    
    for field_number, predictions in field_with_subpatch_labels.items():
        diseased_subpatch_count = np.sum(np.array(predictions) == 0)        #total number of '0' sub-patch labels
        field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

    # Arrange predicted and true labels in lists
    y_pred = []
    y_true = []
    for field_number, predicted_label in field_labels.items():
        if field_number in ground_truth:
            true_label = ground_truth[field_number]
            y_pred.append(predicted_label)
            y_true.append(true_label)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    acc_0 = np.mean(y_pred == y_true)
    recall_0 = recall_score(y_true, y_pred)
    precision_0 = precision_score(y_true, y_pred)
    f1_0 = f1_score(y_true, y_pred)
    f2_0 = fbeta_score(y_true, y_pred, beta=2)


    # Return the metrics with the best assumption: the one with highest recall
    if recall_1 > recall_0:
        return (
            1,
            round(acc_1 * 100, 2),
            round(precision_1 * 100, 2),
            round(recall_1 * 100, 2),
            round(f1_1 * 100, 2),
            round(f2_1 * 100, 2),
        )
    else:
        return (
            0,
            round(acc_0 * 100, 2),
            round(precision_0 * 100, 2),
            round(recall_0 * 100, 2),
            round(f1_0 * 100, 2),
            round(f2_0 * 100, 2),
        )



def assign_field_labels_ae_train(subpatch_coordinates, subpatch_predictions, disease, threshold=0.1):
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
        if disease == 1:
            diseased_subpatch_count = np.sum(np.array(predictions) == 1)
        else:
            diseased_subpatch_count = np.sum(np.array(predictions) == 0)
        field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

    return field_labels


# Utility function to get properly formatted field-labels and co-ords when we don't use dataloaders
