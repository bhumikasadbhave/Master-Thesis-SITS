import time
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


def train_model_ae(model, train_dataloader, test_dataloader, epochs=10, optimizer='Adam', lr=0.001, momentum=0.9, device='mps'):
    """ Vanilla function to train the Autoencoder
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in range(epochs):
        start = time.perf_counter()
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
        end = time.perf_counter()
        print(f"Time taken per epoch: {end - start:.4f} seconds")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")
    return model, epoch_train_losses, epoch_test_losses


### --- AE with Time encodings as channels --- ###
def train_model_ae_te(model, train_dataloader, test_dataloader, out_channels=10, epochs=10, optimizer='Adam', lr=0.001, momentum=0.9, device='mps'):
    """ Vanilla function to train the Autoencoder
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()  
        train_loss = 0.0
        for inputs_cpu, field_numbers in train_dataloader:
            
            inputs = inputs_cpu.to(device)
            latent, reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs[:, :out_channels])
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
                loss = criterion(reconstructed, inputs[:, :out_channels])
                test_loss += loss.item()
        epoch_test_losses.append(test_loss / len(test_dataloader))
        end = time.perf_counter()
        print(f"Time taken per epoch: {end - start:.4f} seconds")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")
    return model, epoch_train_losses, epoch_test_losses



### --- AE with pixel-level Time encodings--- ###
def train_model_ae_te_pixel(model, train_dataloader, test_dataloader, out_channels=10, epochs=10, optimizer='Adam', lr=0.001, momentum=0.9, device='mps'):
    """ Vanilla function to train the Autoencoder
    """
    # Loss and optimizer
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    epoch_train_losses = []
    epoch_test_losses = []
    
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()  
        train_loss = 0.0
        for inputs_cpu, field_numbers, date_embeddings in train_dataloader:
            
            inputs = inputs_cpu.to(device)
            latent, reconstructed = model(inputs, date_embeddings)
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
            for inputs_cpu, field_numbers, date_embeddings in test_dataloader:
                inputs = inputs_cpu.to(device)
                latent, reconstructed = model(inputs, date_embeddings)
                loss = criterion(reconstructed, inputs)
                test_loss += loss.item()
        epoch_test_losses.append(test_loss / len(test_dataloader))
        end = time.perf_counter()
        print(f"Time taken per epoch: {end - start:.4f} seconds")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.6f}, Test Loss: {test_loss / len(test_dataloader):.6f}")
    return model, epoch_train_losses, epoch_test_losses


def extract_features_ae(model, dataloader, temp_embed_pixel=False, device='mps'):
    """ Get latent bottleneck vectors (as features) using only the trained encoder
    """
    features = []
    model.eval()
    with torch.no_grad():
        if not temp_embed_pixel:
            field_numbers_all = []
            for inputs_cpu, field_numbers in dataloader:
                inputs = inputs_cpu.to(device)
                latent, _ = model(inputs)
                features.append(latent.view(latent.size(0), -1))
                field_numbers_all.extend(field_numbers)
        else:
            field_numbers_all = []
            for inputs_cpu, field_numbers, date_emb in dataloader:
                inputs = inputs_cpu.to(device)
                # print(len(field_numbers[0]))
                latent, _ = model(inputs, date_emb)
                features.append(latent.view(latent.size(0), -1))
                field_numbers_all.extend(field_numbers)

    return torch.cat(features), field_numbers_all


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
            
            # VAE losses
            recon_loss = nn.functional.mse_loss(reconstructed, inputs, reduction='sum')    #Reconstruction loss    
            log_var = torch.clamp(log_var, min=-10)  # Prevents numerical instability
            # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())   #ask momo  
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



