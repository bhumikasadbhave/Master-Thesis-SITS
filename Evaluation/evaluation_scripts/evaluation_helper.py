import torch
import numpy as np
import os
import json
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


# Utility Function to get properly formatted field numbers and x,y co-ordinates for the evaluation of subpatches
def get_string_fielddata(patch_coordinates):
    new_coords = []
    for coord in patch_coordinates:
        field_num_coord = '_'.join(map(str, coord))  
        new_coords.append(field_num_coord)
    return new_coords


###  Functions for assigning patch-level labels from sub-patch level labels ---------------------------------------------------

def evaluate_clustering_metrics(subpatch_coordinates, subpatch_predictions, ground_truth_csv_path, threshold=0.5, model_name='Flattened Data', save_pred=False):
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

    #Save y_pred and y_true
    if save_pred==True:
        file_path = config.predictions_path
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[model_name] = {'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()}
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
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



### Functions for getting metrics for patch-level labels directly ---------------------------------------------------

def evaluate_clustering_metrics_patch_level(patch_field_nos, patch_predictions, ground_truth_csv_path):
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
    
    # Prepare the patch level labels: output of this step = {'field#': 0, 'field#': 1, ..} 
    # ie field number and the patch-prediction for that field
    patch_id_labels = {}
    for field_number, label in zip(patch_field_nos, patch_predictions):
        split_field_numbers = field_number.split('_')
        if len(split_field_numbers) == 1:
            patch_id_labels[str(int(float(field_number)))] = label
        else:
            for i in range(len(split_field_numbers)):
                field_number = str(int(float(split_field_numbers[i])))
                patch_id_labels[field_number] = label


    ## Scenario 1: cluster 1=disease
    # No need of aggregation of labels since we use patch-level images directly..
    # output of this step = {'field#': 0, 'field#': 1, .. } => assuming 1=disease
    patch_labels = {}    
    for field_number, prediction in patch_id_labels.items():
        label = prediction                              
        patch_labels[field_number]=1 if prediction==1 else 0     #assuming 1=disease

    # Arrange predicted and true labels in lists
    y_pred = []
    y_true = []
    for field_number, predicted_label in patch_labels.items():
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
    # No need of aggregation of labels since we use patch-level images directly..
    # output of this step = {'field#': 0, 'field#': 1, .. }  => assuming 0=disease
    patch_labels = {}    
    for field_number, prediction in patch_id_labels.items():
        label = prediction          
        patch_labels[field_number]=1 if prediction==0 else 0    #assuming 0=disease

    # Arrange predicted and true labels in lists
    y_pred = []
    y_true = []
    for field_number, predicted_label in patch_labels.items():
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

