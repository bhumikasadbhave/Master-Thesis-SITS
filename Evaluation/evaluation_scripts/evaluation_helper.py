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


def evaluate_clustering_metrics(test_field_labels, ground_truth_csv_path):
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

    precision_per_class = np.diag(cm) / np.sum(cm, axis=0, where=(np.sum(cm, axis=0) != 0))
    recall_per_class = np.diag(cm) / np.sum(cm, axis=1, where=(np.sum(cm, axis=1) != 0))
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    fmi = fowlkes_mallows_score(y_true, mapped_preds)

    # precision = precision_score(y_true, mapped_preds)
    # f1 = f1_score(y_true, mapped_preds)
    # recall = recall_score(y_true, mapped_preds)
    f2_score = fbeta_score(y_true, mapped_preds, beta=2)

    return acc, precision_per_class, recall_per_class, f1_per_class, f2_score






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


################################ New functions for assigning labels #######################################

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


def assign_field_labels_ae_eval(subpatch_coordinates, subpatch_predictions, threshold=0.1):
    """
    Assign patch/field-level labels based on sub-patch predictions
    Returns: field_labels: Dictionary {field_number: field_label}
    Returns two versions of patch-level labels (one assuming 1=diseased, one assuming 0=diseased)
    """
    field_dict = {}
    for field_number, prediction in zip(subpatch_coordinates, subpatch_predictions):
        if field_number not in field_dict:
            field_dict[field_number] = []
        field_dict[field_number].append(prediction)

    field_labels_1 = {}  # Assuming "1" = diseased
    field_labels_0 = {}  # Assuming "0" = diseased

    for field_number, predictions in field_dict.items():
        num_subpatches = len(predictions)
        num_diseased_1 = np.sum(np.array(predictions) == 1)
        num_diseased_0 = np.sum(np.array(predictions) == 0)

        # Thresholding for both possible mappings
        field_labels_1[field_number] = 1 if num_diseased_1 >= (threshold * num_subpatches) else 0
        field_labels_0[field_number] = 1 if num_diseased_0 >= (threshold * num_subpatches) else 0

    return field_labels_1, field_labels_0


def evaluate_best_mapping_new(subpatch_coordinates, subpatch_predictions, ground_truth_csv_path, threshold=0.1):
    """
    Evaluate clustering metrics using both mappings (0=diseased or 1=diseased) and choose the best one using Hungarian algorithm
    """
    disease = 0
    field_labels_1, field_labels_0 = assign_field_labels_ae_eval(subpatch_coordinates, subpatch_predictions, threshold)

    acc_1, precision_1, recall_1, f1_1, f2_1 = evaluate_clustering_metrics_wo_hungarian(field_labels_1, ground_truth_csv_path)
    acc_0, precision_0, recall_0, f1_0, f2_0 = evaluate_clustering_metrics_wo_hungarian(field_labels_0, ground_truth_csv_path)

    if acc_1 > acc_0:
        print("Selected 1 = Diseased mapping")
        disease = 1
        return disease, acc_1, precision_1, recall_1, f1_1, f2_1
    else:
        print("Selected 0 = Diseased mapping")
        disease = 0
        return disease, acc_0, precision_0, recall_0, f1_0, f2_0



def evaluate_clustering_metrics_wo_hungarian(test_field_labels, ground_truth_csv_path):
    """
    Evaluate clustering accuracy (ACC), precision, recall, and F1-score per class.
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

    acc = np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred)

    precision_per_class = np.diag(cm) / np.sum(cm, axis=0, where=(np.sum(cm, axis=0) != 0))
    recall_per_class = np.diag(cm) / np.sum(cm, axis=1, where=(np.sum(cm, axis=1) != 0))
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
    fmi = fowlkes_mallows_score(y_true, y_pred)

    # precision = precision_score(y_true, mapped_preds)
    # f1 = f1_score(y_true, mapped_preds)
    # recall = recall_score(y_true, mapped_preds)
    f2_score = fbeta_score(y_true, y_pred, beta=2)

    return acc, precision_per_class, recall_per_class, f1_per_class, f2_score





# ################################ functions for DCEC #######################################

# def assign_field_labels_clustering(subpatch_coordinates, subpatch_predictions, threshold=0.1):
#     """
#     Assign patch/field-level labels based on sub-patch clustering predictions.
#     Returns: field_labels: Dictionary {field_number: field_label}
#     """
#     field_dict = {}
    
#     # Assign each sub-patch to the predicted cluster (highest probability in q)
#     for field_number, prediction in zip(subpatch_coordinates, subpatch_predictions):
#         # Find the index of the highest cluster assignment probability
#         cluster_label = prediction  # Using the cluster with max probability
#         if field_number not in field_dict:
#             field_dict[field_number] = []
#         field_dict[field_number].append(cluster_label)

#     field_labels = {}    
#     for field_number, predictions in field_dict.items():
#         # Count the number of sub-patches that belong to the diseased cluster (cluster 1, for example)
#         diseased_subpatch_count = np.sum(np.array(predictions) == 1)  # Assuming cluster 1 is the diseased cluster
        
#         # Assign the label based on the threshold
#         field_labels[field_number] = 1 if diseased_subpatch_count >= (threshold * len(predictions)) else 0

#     return field_labels