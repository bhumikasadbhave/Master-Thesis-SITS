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



def train_mae(model, dataloader, optimizer, device, criterion=torch.nn.MSELoss()):
    model.train()
    total_loss = 0.0
    for images, _ in dataloader:  
        images = images.to(device)
        optimizer.zero_grad()
        pred, mask, loss = model(images, mask_ratio=0.75)  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model_mae(model, train_dataloader, test_dataloader, epochs=10, masking_ratio=0.75, optimizer='Adam', lr=0.001, momentum=0.9, device='cuda'):
    """Train a Masked Autoencoder (MAE)."""

    # criterion = nn.MSELoss()
    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("Unsupported optimizer. Choose 'Adam' or 'SGD'.")

    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(epochs):
        model.train()  
        train_loss = 0.0

        for inputs_cpu, field_numbers, timestamps in train_dataloader:
            inputs, timestamps = inputs_cpu.to(device), timestamps.to(device)
            # print(inputs.shape)
            optimizer.zero_grad() 
            loss, pred, mask, latent = model(inputs, timestamps, mask_ratio=masking_ratio) 
            # print(latent.shape) 
            # print("Loss requires grad:", loss.requires_grad)
            # print("Pred requires grad:", pred.requires_grad)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()        
        epoch_train_losses.append(train_loss / len(train_dataloader))

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs_cpu, field_numbers, timestamps in test_dataloader: 
                inputs, timestamps = inputs_cpu.to(device), timestamps.to(device)

                loss, pred, mask, latent = model(inputs, timestamps, mask_ratio=masking_ratio) 
                test_loss += loss.item()
        epoch_test_losses.append(test_loss / len(test_dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")

    return model, epoch_train_losses, epoch_test_losses


def extract_latent_features_mae(model, dataloader, device):
    model.eval()
    latents = []
    field_numbers_all = []
    with torch.no_grad():
        for imgs, field_numbers, timestamps in dataloader:
            imgs, timestamps = imgs.to(device), timestamps.to(device)
            _, _, _, latent = model(imgs, timestamps)  # Get latent
            latents.append(latent[:, 0, :].cpu())  # Take only the CLS token
            field_numbers_all.extend(field_numbers)
    return torch.cat(latents, dim=0), field_numbers_all


