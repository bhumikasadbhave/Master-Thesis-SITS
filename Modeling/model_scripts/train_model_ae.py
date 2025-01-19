import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score



def train_model_ae(model, dataloader, epochs=10, lr=0.001, device='mps'):
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs_cpu, field_numbers in dataloader:
            inputs = inputs_cpu.to(device)
            # inputs, field_numbers = batch                   # Extract inputs
            optimizer.zero_grad()
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss/len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    return model, epoch_losses


def extract_features_ae(model, dataloader, device='mps'):
    features = []
    field_numbers_all = []
    model.eval()
    with torch.no_grad():
        for inputs_cpu, field_numbers in dataloader:
            inputs = inputs_cpu.to(device)
            latent, _ = model(inputs)
            features.append(latent.view(latent.size(0), -1))
            field_numbers_all.extend(field_numbers)
    return torch.cat(features), field_numbers_all



def apply_kmeans(features, n_clusters=2, random_state=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    predictions = kmeans.fit(features.cpu().numpy())
    return kmeans


def apply_agglomerative_clustering(features, n_clusters=2):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    predictions = agg_clustering.fit_predict(features.cpu().numpy())
    return agg_clustering, predictions


def apply_dbscan(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    predictions = dbscan.fit_predict(features.cpu().numpy())
    return dbscan, predictions


# Evaluation function only for TEST
def get_gt_and_pred_aligned(field_numbers, labels, gt_path):

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

    gt_aligned = []
    pred_aligned = []
    for field_number, predicted_label in field_labels.items():
        if field_number in gt_mapping:
            gt_aligned.append(1 if gt_mapping[field_number] == 'yes' else 0)
            pred_aligned.append(predicted_label)

    return gt_aligned, pred_aligned


    
# Evaluation function only for TEST
def evaluate_clustering_metrics(gt_aligned, pred_aligned):
    accuracy = accuracy_score(gt_aligned, pred_aligned)
    ari = adjusted_rand_score(gt_aligned, pred_aligned)
    nmi = normalized_mutual_info_score(gt_aligned, pred_aligned)
    fmi = fowlkes_mallows_score(gt_aligned, pred_aligned)

    metrics = {
        "Accuracy": accuracy,
        "Adjusted Rand Index (ARI)": ari,
        "Normalized Mutual Information (NMI)": nmi,
        "Fowlkes-Mallows Index (FMI)": fmi,
    }
    return metrics

