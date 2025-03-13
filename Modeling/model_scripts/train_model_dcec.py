import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np
from torch.optim import Adam

def pretrain_autoencoder(model, train_dataloader, test_dataloader, epochs=10, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = Adam(model.autoencoder.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, _ in train_dataloader:
            x_batch = x_batch.to(device)
            z, x_reconstructed = model.autoencoder(x_batch)

            loss = criterion(x_reconstructed, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Test Loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, _ in test_dataloader:
                x_batch = x_batch.to(device)
                z, x_reconstructed = model.autoencoder(x_batch)
                test_loss += criterion(x_reconstructed, x_batch).item()
        
        test_loss /= len(test_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} - Autoencoder Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    print("Autoencoder Pretraining Complete!")


def initialize_clusters(model, dataloader, device='cuda'):
    model.to(device)
    model.eval()
    z_values = []
    
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            z, _ = model.autoencoder(x_batch)
            z_values.append(z.cpu().numpy())
    
    z_values = np.concatenate(z_values, axis=0)
    kmeans = KMeans(n_clusters=model.clustering_layer.n_clusters, n_init=20, random_state=42)
    cluster_centers = kmeans.fit(z_values).cluster_centers_
    
    model.clustering_layer.cluster_centers.data = torch.tensor(cluster_centers, dtype=torch.float32, device=device)
    print("Cluster Centers Initialized!")


def target_distribution(q):
    """Update the target distribution P."""
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T  # Normalize across samples



def train_dcec(model, train_dataloader, test_dataloader, epochs=50, lr=1e-3, gamma=0.1, T=10, device='cuda'):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion_reconstruction = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_recon_loss, train_clustering_loss, train_total_loss = 0, 0, 0

        for x_batch, _ in train_dataloader:
            x_batch = x_batch.to(device)
            q, x_reconstructed = model(x_batch)

            # Compute Target Distribution P (every T epochs)
            if epoch % T == 0:
                target_q = (q ** 2) / q.sum(0, keepdim=True)
                target_q = target_q / target_q.sum(1, keepdim=True)

            # Compute Losses
            recon_loss = criterion_reconstruction(x_reconstructed, x_batch)
            clustering_loss = F.kl_div(q.log(), target_q, reduction="batchmean")
            total_loss = recon_loss + gamma * clustering_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_recon_loss += recon_loss.item()
            train_clustering_loss += clustering_loss.item()
            train_total_loss += total_loss.item()

        train_recon_loss /= len(train_dataloader)
        train_clustering_loss /= len(train_dataloader)
        train_total_loss /= len(train_dataloader)

        # Compute Test Loss
        model.eval()
        test_recon_loss, test_clustering_loss, test_total_loss = 0, 0, 0

        with torch.no_grad():
            for x_batch, _ in test_dataloader:
                x_batch = x_batch.to(device)
                q, x_reconstructed = model(x_batch)

                # Update Target P (for test loss computation)
                target_q = (q ** 2) / q.sum(0, keepdim=True)
                target_q = target_q / target_q.sum(1, keepdim=True)

                recon_loss = criterion_reconstruction(x_reconstructed, x_batch)
                clustering_loss = F.kl_div(q.log(), target_q, reduction="batchmean")
                total_loss = recon_loss + gamma * clustering_loss

                test_recon_loss += recon_loss.item()
                test_clustering_loss += clustering_loss.item()
                test_total_loss += total_loss.item()

        test_recon_loss /= len(test_dataloader)
        test_clustering_loss /= len(test_dataloader)
        test_total_loss /= len(test_dataloader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_total_loss:.4f} (Recon: {train_recon_loss:.4f}, Clust: {train_clustering_loss:.4f}) | "
              f"Test Loss: {test_total_loss:.4f} (Recon: {test_recon_loss:.4f}, Clust: {test_clustering_loss:.4f})")

    print("DCEC Training Complete!")



### DCEC Functions ### 

def train_dcec_old(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, momentum=0.9, alpha=1.0, device='mps'):
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
            target_q = (q ** 2) / (q.sum(0, keepdim=True) + 1e-10)
            target_q = target_q / (target_q.sum(1, keepdim=True) + 1e-10)
            
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
                target_q = (q ** 2) / (q.sum(0, keepdim=True) + 1e-10)
                target_q = target_q / (target_q.sum(1, keepdim=True) + 1e-10)

                
                reconstruction_loss = mse_loss(reconstructed, inputs)
                clustering_loss = kl_loss(q.log(), target_q.detach())

                total_test_reconstruction_loss += reconstruction_loss.item()
                total_test_clustering_loss += clustering_loss.item()
        
        # Calculate average test losses
        avg_test_reconstruction_loss = total_test_reconstruction_loss / len(test_dataloader)
        avg_test_clustering_loss = total_test_clustering_loss / len(test_dataloader)

        # Print losses for both train and test sets
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Reconstruction Loss: {avg_train_reconstruction_loss:.6f}, Clustering Loss: {avg_train_clustering_loss:.6f}")
        print(f"  Test  - Reconstruction Loss: {avg_test_reconstruction_loss:.6f}, Clustering Loss: {avg_test_clustering_loss:.6f}")
        
        # Store losses for each epoch
        epoch_losses['train_reconstruction_loss'].append(avg_train_reconstruction_loss)
        epoch_losses['train_clustering_loss'].append(avg_train_clustering_loss)
        epoch_losses['test_reconstruction_loss'].append(avg_test_reconstruction_loss)
        epoch_losses['test_clustering_loss'].append(avg_test_clustering_loss)

    # Return the model and all epoch losses
    return model, epoch_losses



def evaluate_dcec_old(model, eval_dataloader, device='mps'):
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


