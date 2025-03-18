import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


### --- Functions for Evaluating Patch level data (for Preliminary tests) --- ###

def get_accuracy(field_numbers, labels, gt_path):
    """ This function is used for computing classification accuracy of patch-level data without Hungarian Algorithm
        There is a separate function because we don't need to convert sub-patch level labels to 
        patch-level labels using threshold.
    """
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

    correct = 0
    total = len(field_labels)
    for field_number, predicted_label in field_labels.items():
        ground_truth = gt_mapping.get(field_number, None)
        if (predicted_label == 1 and ground_truth == 'yes') or (predicted_label == 0 and ground_truth == 'no'):
            correct += 1

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


def get_clustering_accuracy(field_numbers, labels, gt_path):
    """
    Compute clustering accuracy, precision, recall, and F1-score per class using the Hungarian algorithm 
    for optimal matching of clusters
    This function is used for evaluation of patch-level data.
    """
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
                field_labels[int(float(n))] = labels[i]
        else:
            field_labels[int(float(number))] = labels[i]

    y_true = []
    y_pred = []
    for field_number, predicted_label in field_labels.items():
        if field_number in gt_mapping:
            y_true.append(1 if gt_mapping[field_number] == 'yes' else 0)
            y_pred.append(predicted_label)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique clusters and labels
    unique_clusters = np.unique(y_pred)
    unique_labels = np.unique(y_true)
    num_clusters = len(unique_clusters)
    num_labels = len(unique_labels)

    # Cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((num_clusters, num_labels))
    for i, cluster in enumerate(unique_clusters):
        for j, label in enumerate(unique_labels):
            cost_matrix[i, j] = -np.sum((y_pred == cluster) & (y_true == label))  # Negative for maximization

    # Assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Best mapping
    mapping = {unique_clusters[row]: unique_labels[col] for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([mapping[pred] for pred in y_pred])

    acc = np.mean(mapped_preds == y_true)
    cm = confusion_matrix(y_true, mapped_preds, labels=unique_labels)
    precision_per_class = np.diag(cm) / np.sum(cm, axis=0, where=(np.sum(cm, axis=0) != 0))
    recall_per_class = np.diag(cm) / np.sum(cm, axis=1, where=(np.sum(cm, axis=1) != 0))
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)

    return acc, precision_per_class, recall_per_class, f1_per_class
