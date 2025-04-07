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


def train_model_ae_old(model, dataloader, epochs=10, lr=0.001, device='mps'):
    """ Function to train the autoencoder without test set :( -> remove
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs_cpu, field_numbers in dataloader:
            inputs = inputs_cpu.to(device)
            # inputs, field_numbers = batch                   
            optimizer.zero_grad()
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss/len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    return model, epoch_losses


def train_model_ae(model, train_dataloader, test_dataloader, epochs=10, optimizer='Adam', lr=0.001, momentum=0.9, weight_decay=0.01, device='mps'):
    """ Vanilla function to train the Autoencoder
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in range(epochs):
        model.train()  
        train_loss = 0.0
        for inputs_cpu, field_numbers in train_dataloader:
            
            inputs = inputs_cpu.to(device)
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()        
        epoch_train_losses.append(train_loss / len(train_dataloader))
        
        # Evaluate on the test set
        model.eval()  
        test_loss = 0.0
        with torch.no_grad():  
            for inputs_cpu, field_numbers in test_dataloader:
                inputs = inputs_cpu.to(device)
                latent, reconstructed = model(inputs)
                loss = criterion(reconstructed, inputs)
                test_loss += loss.item()
        epoch_test_losses.append(test_loss / len(test_dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")
    return model, epoch_train_losses, epoch_test_losses


def extract_features_ae(model, dataloader, device='mps'):
    """ Get latent bottleneck vectors (as features) using only the trained encoder
    """
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


def get_string_fielddata(patch_coordinates):
    new_coords = []
    for coord in patch_coordinates:
        field_num_coord = '_'.join(map(str, coord))  
        new_coords.append(field_num_coord)
    return new_coords


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

# Evaluation function (for patch-level data)
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


# Evaluation function only for evaluation set
def evaluate_clustering_metrics_old(gt_aligned, pred_aligned):
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


### --- VAE functions --- ###

def train_model_vae(model, train_dataloader, test_dataloader, epochs=10, lr=0.001,optimizer='Adam', momentum=0.9, device='mps'):

    # Loss: L2 loss (Squared Sum of Errors)
    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    epoch_train_recon_losses = []
    epoch_train_kl_losses = []
    epoch_test_recon_losses = []
    epoch_test_kl_losses = []

    for epoch in range(epochs):
        model.train()
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        for inputs_cpu, field_numbers in train_dataloader:
            
            inputs = inputs_cpu.to(device)
            mu, log_var, z, reconstructed = model(inputs)
            
            # Compute VAE losses
            # recon_loss = criterion(reconstructed, inputs)
            # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            recon_loss = nn.functional.mse_loss(reconstructed, inputs, reduction='sum')    #Reconstruction loss    
            log_var = torch.clamp(log_var, min=-10)  # Prevents numerical instability
            # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())   ask momo  
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()    #mean over batch     
            loss = recon_loss + kl_divergence
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_divergence.item()
        
        epoch_train_recon_losses.append(train_recon_loss / len(train_dataloader))
        epoch_train_kl_losses.append(train_kl_loss / len(train_dataloader))
        
        # Evaluation on test
        model.eval()
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        
        with torch.no_grad():
            for inputs_cpu, field_numbers in test_dataloader:
                inputs = inputs_cpu.to(device)

                mu, log_var, z, reconstructed = model(inputs)
                
                # recon_loss = criterion(reconstructed, inputs)
                # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                recon_loss = nn.functional.mse_loss(reconstructed, inputs, reduction='sum')    #Reconstruction loss    
                log_var = torch.clamp(log_var, min=-10)  # Prevents numerical instability
                # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()            
                loss = recon_loss + kl_divergence
                
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_divergence.item()
        
        epoch_test_recon_losses.append(test_recon_loss / len(test_dataloader))
        epoch_test_kl_losses.append(test_kl_loss / len(test_dataloader))
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Recon Loss: {train_recon_loss / len(train_dataloader):.4f}, Train KL Loss: {train_kl_loss / len(train_dataloader):.4f}")
        print(f"  Test Recon Loss: {test_recon_loss / len(test_dataloader):.4f}, Test KL Loss: {test_kl_loss / len(test_dataloader):.4f}")
    return model, epoch_train_recon_losses, epoch_train_kl_losses, epoch_test_recon_losses, epoch_test_kl_losses



def extract_features_vae(model, dataloader, device='mps'):
    features = []
    field_numbers_all = []
    model.eval()
    with torch.no_grad():
        for inputs_cpu, field_numbers in dataloader:
            inputs = inputs_cpu.to(device)
            mu, log_var, z, reconstructed = model(inputs)
            features.append(z.view(z.size(0), -1))
            field_numbers_all.extend(field_numbers)
    return torch.cat(features), field_numbers_all



