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


def train_model_ae(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, device='mps'):
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_train_losses = []
    epoch_test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()  
        train_loss = 0.0
        for inputs_cpu, field_numbers in train_dataloader:
            inputs = inputs_cpu.to(device)
            optimizer.zero_grad()
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
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
        
        # Print the loss for both train and test sets
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Test Loss: {test_loss / len(test_dataloader):.4f}")
    
    return model, epoch_train_losses, epoch_test_losses


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


# Evaluation function only for eval
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


### VAE functions ###

def train_model_vae(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, device='mps'):
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_train_losses = []
    epoch_test_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs_cpu, field_numbers in train_dataloader:
            inputs = inputs_cpu.to(device)
            optimizer.zero_grad()
            
            mu, log_var, z, reconstructed = model(inputs)
            
            # Compute VAE loss
            recon_loss = criterion(reconstructed, inputs)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_divergence
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        epoch_train_losses.append(train_loss / len(train_dataloader))
        
        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs_cpu, field_numbers in test_dataloader:
                inputs = inputs_cpu.to(device)
                mu, log_var, z, reconstructed = model(inputs)
                recon_loss = criterion(reconstructed, inputs)
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_divergence
                test_loss += loss.item()
        
        epoch_test_losses.append(test_loss / len(test_dataloader))
        
        # Print the loss for both train and test sets
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Test Loss: {test_loss / len(test_dataloader):.4f}")
    
    return model, epoch_train_losses, epoch_test_losses

def extract_features_vae(model, dataloader, device='mps'):
    features = []
    field_numbers_all = []
    model.eval()
    with torch.no_grad():
        for inputs_cpu, field_numbers in dataloader:
            inputs = inputs_cpu.to(device)
            mu, log_var, z, _ = model(inputs)
            features.append(z.view(z.size(0), -1))
            field_numbers_all.extend(field_numbers)
    return torch.cat(features), field_numbers_all



### DCEC Functions ### 

# def train_dcec(model, train_dataloader, epochs=10, lr=0.001, alpha=1.0, device='mps'):
#     model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     mse_loss = nn.MSELoss()
#     kl_loss = nn.KLDivLoss(reduction='batchmean')

#     for epoch in range(epochs):
#         model.train()
#         total_reconstruction_loss = 0.0
#         total_clustering_loss = 0.0
        
#         for inputs, _ in train_dataloader:
#             inputs = inputs.to(device)
#             optimizer.zero_grad()
            
#             q, reconstructed = model(inputs)
#             target_q = (q ** 2) / q.sum(0)
#             target_q = target_q / target_q.sum(1, keepdim=True)
            
#             reconstruction_loss = mse_loss(reconstructed, inputs)
#             clustering_loss = kl_loss(q.log(), target_q.detach())

#             loss = reconstruction_loss + alpha * clustering_loss
#             loss.backward()
#             optimizer.step()
            
#             total_reconstruction_loss += reconstruction_loss.item()
#             total_clustering_loss += clustering_loss.item()
        
#         print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {total_reconstruction_loss / len(train_dataloader):.4f}, Clustering Loss: {total_clustering_loss / len(train_dataloader):.4f}")

#     return model

def train_dcec(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, alpha=1.0, device='mps'):
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    # To store the epoch losses for both train and test
    epoch_losses = {
        'train_reconstruction_loss': [],
        'train_clustering_loss': [],
        'test_reconstruction_loss': [],
        'test_clustering_loss': []
    }

    for epoch in range(epochs):
        # Train phase
        model.train()
        total_train_reconstruction_loss = 0.0
        total_train_clustering_loss = 0.0
        
        for inputs, _ in train_dataloader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            q, reconstructed = model(inputs)
            target_q = (q ** 2) / q.sum(0)
            target_q = target_q / target_q.sum(1, keepdim=True)
            
            reconstruction_loss = mse_loss(reconstructed, inputs)
            clustering_loss = kl_loss(q.log(), target_q.detach())

            loss = reconstruction_loss + alpha * clustering_loss
            loss.backward()
            optimizer.step()
            
            total_train_reconstruction_loss += reconstruction_loss.item()
            total_train_clustering_loss += clustering_loss.item()
        
        # Calculate average train losses
        avg_train_reconstruction_loss = total_train_reconstruction_loss / len(train_dataloader)
        avg_train_clustering_loss = total_train_clustering_loss / len(train_dataloader)

        # Test phase
        model.eval()
        total_test_reconstruction_loss = 0.0
        total_test_clustering_loss = 0.0
        
        with torch.no_grad():  
            for inputs, _ in test_dataloader:
                inputs = inputs.to(device)
                
                q, reconstructed = model(inputs)
                target_q = (q ** 2) / q.sum(0)
                target_q = target_q / target_q.sum(1, keepdim=True)
                
                reconstruction_loss = mse_loss(reconstructed, inputs)
                clustering_loss = kl_loss(q.log(), target_q.detach())

                total_test_reconstruction_loss += reconstruction_loss.item()
                total_test_clustering_loss += clustering_loss.item()
        
        # Calculate average test losses
        avg_test_reconstruction_loss = total_test_reconstruction_loss / len(test_dataloader)
        avg_test_clustering_loss = total_test_clustering_loss / len(test_dataloader)

        # Print losses for both train and test sets
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Reconstruction Loss: {avg_train_reconstruction_loss:.4f}, Clustering Loss: {avg_train_clustering_loss:.4f}")
        print(f"  Test  - Reconstruction Loss: {avg_test_reconstruction_loss:.4f}, Clustering Loss: {avg_test_clustering_loss:.4f}")
        
        # Store losses for each epoch
        epoch_losses['train_reconstruction_loss'].append(avg_train_reconstruction_loss)
        epoch_losses['train_clustering_loss'].append(avg_train_clustering_loss)
        epoch_losses['test_reconstruction_loss'].append(avg_test_reconstruction_loss)
        epoch_losses['test_clustering_loss'].append(avg_test_clustering_loss)

    # Return the model and all epoch losses
    return model, epoch_losses



def evaluate_dcec(model, eval_dataloader, device='mps'):
    model.to(device)
    model.eval() 
    
    latent_features = []
    cluster_assignments = []
    field_numbers_all = []
    
    with torch.no_grad():
        for inputs_cpu, field_numbers in eval_dataloader:
            inputs = inputs_cpu.to(device)
            
            # Perform a forward pass to get the latent features (z) and cluster assignments (q)
            q, _ = model(inputs)
            
            # Append the cluster assignments (for analysis) and latent features
            latent_features.append(q.cpu())  # We store the cluster assignments for analysis
            cluster_assignments.append(q.argmax(dim=1).cpu())  # Assign to the cluster with the highest probability
            field_numbers_all.extend(field_numbers)
    
    # Concatenate the latent features (q)
    latent_features = torch.cat(latent_features, dim=0)
    cluster_assignments = torch.cat(cluster_assignments, dim=0)
    
    return latent_features, cluster_assignments, field_numbers_all
