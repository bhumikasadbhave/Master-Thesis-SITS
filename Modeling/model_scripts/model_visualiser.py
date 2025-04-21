from ast import List, Tuple
from ctypes import Union
import math
import random
import warnings
from typing import Optional
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import config


def plot_loss(train_loss, test_loss, title="Training vs Test Loss"):
    """
    Plots training and test loss over epochs.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.2)
    plt.show()


def plot_loss_log_scale(train_loss, test_loss, title="Training vs Test Loss (Log Scale)"):
    """
    Plots training and test loss over epochs with a logarithmic scale on y-axis.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-', color='red')

    plt.yscale('log')  # Set log scale for y-axis
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)  # Grid for log scale
    plt.show()


def normalize_for_display(image):
    """ Adjusts brightness & contrast for visualization. """
    image = image - image.min()  # Shift to [0, max]
    image = image / (image.max() - image.min())  # Normalize to [0,1]
    return image


#### ----------------------------------- Functions for plotting reconstructions ------------------------------------ ####

def plot_reconstructed_subpatches(model, dataloader, num_images=5, temporal_index=0, device='mps'):
    """Plots original and reconstructed subpatches side by side"""
    
    model.eval()
    indices = random.sample(range(config.batch_size), num_images)
    
    with torch.no_grad():
        for inputs_cpu, _ in dataloader:
            inputs = inputs_cpu.to(device)
            _, reconstructed = model(inputs)

            # Move tensors to CPU for visualization
            inputs = inputs.cpu()
            reconstructed = reconstructed.cpu()
            
            fig, axes = plt.subplots(num_images, 2, figsize=(8, 2 * num_images))
            
            for i, index in enumerate(indices):
                # Extract the first 3 channels from the first temporal image
                original_img = inputs[index, :3, temporal_index, :, :].permute(1, 2, 0)  # (H, W, 3)
                original_img = normalize_for_display(original_img)
                reconstructed_img = reconstructed[index, :3, temporal_index, :, :].permute(1, 2, 0)  # (H, W, 3)
                reconstructed_img = normalize_for_display(reconstructed_img)

                # Plot original image
                axes[i, 0].imshow(original_img.numpy())
                axes[i, 0].set_title("Original")
                axes[i, 0].axis("off")

                # Plot reconstructed image
                axes[i, 1].imshow(reconstructed_img.numpy())
                axes[i, 1].set_title("Reconstructed")
                axes[i, 1].axis("off")

            plt.show()
            break  # Only take the first batch


def plot_reconstructed_subpatches_temporal(model, dataloader, num_images=5, device='cuda', model_type='ae'):
    """Plots all temporal images (original vs. reconstructed) for selected patches"""
    
    model.eval()
    temporal_len = None  
    indices = random.sample(range(config.batch_size), num_images)

    with torch.no_grad():
        if model_type in ['ae','vae']:
            for inputs_cpu, _ in dataloader:

                inputs = inputs_cpu.to(device)
                if model_type=='ae':
                    _, reconstructed = model(inputs)
                elif model_type=='vae':
                    _,_,_, reconstructed = model(inputs)

                inputs = inputs.cpu()
                reconstructed = reconstructed.cpu()
                temporal_len = inputs.shape[2]  

                for i, index in enumerate(indices):
                    fig, axes = plt.subplots(2, temporal_len, figsize=(3 * temporal_len, 6))

                    for t in range(temporal_len):
                        # Original
                        original_img = inputs[index, :3, t, :, :].permute(1, 2, 0)  # (H, W, 3)
                        original_img = normalize_for_display(original_img)

                        # Reconstructed
                        reconstructed_img = reconstructed[index, :3, t, :, :].permute(1, 2, 0)
                        reconstructed_img = normalize_for_display(reconstructed_img)

                        axes[0, t].imshow(original_img.numpy())
                        axes[0, t].set_title(f"Original T{t}")
                        axes[0, t].axis("off")

                        axes[1, t].imshow(reconstructed_img.numpy())
                        axes[1, t].set_title(f"Reconstructed T{t}")
                        axes[1, t].axis("off")

                    plt.tight_layout()
                    plt.show()

                break  # Only take the first batch
        
        elif model_type in ['ae_te']:
            for inputs_cpu, _, date_embeddings in dataloader:

                inputs = inputs_cpu.to(device)
                _, reconstructed = model(inputs, date_embeddings)
                inputs = inputs.cpu()
                reconstructed = reconstructed.cpu()
                temporal_len = inputs.shape[2]  

                for i, index in enumerate(indices):
                    fig, axes = plt.subplots(2, temporal_len, figsize=(3 * temporal_len, 6))

                    for t in range(temporal_len):
                        # Original
                        original_img = inputs[index, :3, t, :, :].permute(1, 2, 0)  # (H, W, 3)
                        original_img = normalize_for_display(original_img)

                        # Reconstructed
                        reconstructed_img = reconstructed[index, :3, t, :, :].permute(1, 2, 0)
                        reconstructed_img = normalize_for_display(reconstructed_img)

                        axes[0, t].imshow(original_img.numpy())
                        axes[0, t].set_title(f"Original T{t}")
                        axes[0, t].axis("off")

                        axes[1, t].imshow(reconstructed_img.numpy())
                        axes[1, t].set_title(f"Reconstructed T{t}")
                        axes[1, t].axis("off")

                    plt.tight_layout()
                    plt.show()

                break  # Only take the first batch


def plot_reconstructed_patches_temporal(model, dataloader, old_images, num_fields=5, device='cuda', model_type='ae'):
    """Map patch-level reconstructions onto patch-level images"""

    model.eval()
    recon_dict = {}         #format: field_number → [(x, y, recon_patch),..]

    #Collect all reconstructions grouped by field numbers
    if model_type in ['ae','vae']:
        for idx in range(len(dataloader.dataset)):
            inputs, patch_id_xy = dataloader.dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)

            with torch.no_grad():
                if model_type=='ae':
                    _, outputs = model(inputs)
                elif model_type == 'vae':
                    _, _, _, outputs = model(inputs)
                    
            outputs = outputs.cpu().squeeze(0)          # [C, T, 4, 4]
            id_coords = patch_id_xy.split('_')
            x, y = int(id_coords[-2]), int(id_coords[-1])
            field_number = '_'.join(id_coords[:-2])

            if field_number not in recon_dict:
                recon_dict[field_number] = []
            recon_dict[field_number].append((x, y, outputs[:3])) 
    
    elif model_type in ['ae_te']:
        for idx in range(len(dataloader.dataset)):
            inputs, patch_id_xy, date_emb = dataloader.dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            date_emb = torch.tensor(date_emb).unsqueeze(0).to(device)

            with torch.no_grad():
                _, outputs = model(inputs, date_emb)
                    
            outputs = outputs.cpu().squeeze(0)          # [C, T, 4, 4]
            id_coords = patch_id_xy.split('_')
            x, y = int(id_coords[-2]), int(id_coords[-1])
            field_number = '_'.join(id_coords[:-2])

            if field_number not in recon_dict:
                recon_dict[field_number] = []
            recon_dict[field_number].append((x, y, outputs[:3])) 

    #Visualise random field numbers
    chosen_fields = random.sample(list(recon_dict.keys()), num_fields)

    for field_number in chosen_fields:
        patch_list = recon_dict[field_number]
        original_temporal = old_images[field_number]        #[7, 64, 64, 12]
        mask = original_temporal[0][:, :, 0] != 0           #mask for actual sugarbeet field pixels
        recon_image = np.zeros((64, 64, 3, 7))

        for (x, y, patch_recon) in patch_list:
            for t in range(7):
                patch_np = patch_recon[:, t].permute(1, 2, 0).numpy()  # [4, 4, 3]
                recon_image[y:y+4, x:x+4, :, t] = patch_np
        for t in range(7):
            for c in range(3):
                recon_image[:, :, c, t] *= mask                 #Mask reconstruction

        fig, axs = plt.subplots(2, 7, figsize=(21, 6))          #PLOT
        fig.suptitle(f'Field {field_number}', fontsize=16)
        for t in range(7):
            #Original image
            axs[0, t].imshow(normalize_for_display(original_temporal[t][:, :, :3]))  
            axs[0, t].set_title(f'Original T{t}')
            axs[0, t].axis('off')
            #Reconstruction
            axs[1, t].imshow(normalize_for_display(recon_image[:, :, :, t]))
            axs[1, t].set_title(f'Reconstruction T{t}')
            axs[1, t].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) 
        plt.show()


#### ----------------------------------- Functions for plotting Losses ------------------------------------ #### 

def plot_all_models_loss_curves(model_names, avg_train_losses, avg_test_losses):
    """ Plots the train and test loss curves for all models. """
    epochs = list(range(1, len(avg_train_losses[0]) + 1))

    plt.figure(figsize=(14, 6))

    # Train Loss subplot
    plt.subplot(1, 2, 1)
    for model_name, train_loss in zip(model_names, avg_train_losses):
        plt.plot(epochs, train_loss, label=model_name)
    plt.title("Train Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Test Loss subplot
    plt.subplot(1, 2, 2)
    for model_name, test_loss in zip(model_names, avg_test_losses):
        plt.plot(epochs, test_loss, label=model_name)
    plt.title("Test Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



#### ----------------------------------- Functions for plotting MAE reconstructions ------------------------------------ ####

def visualize_temporal_reconstructions_mae(model, dataloader, device, num_images=5, T=3):
    """ Visualizes reconstructed images over multiple temporal frames.
    """
    model.eval()
    imgs_list, recon_list = [], []

    with torch.no_grad():
        for imgs, fn, timestamps in dataloader:
            imgs, timestamps = imgs.to(device), torch.stack(timestamps).to(device)
            loss, pred, mask, latent = model(imgs, timestamps)
            print('recon values',pred.min(),pred.max())

            # print(f"Pred shape: {pred.shape}")  # (N, T*L, patch_size^2 * C)
            # print("Images shape: ", imgs.shape)  # (N, C, T, H, W)

            # Reshape pred from (N, T*L, P^2*C) to (N, T, L, P^2*C)
            N, L_total, O = pred.shape
            L = L_total // T  # Number of patches per image -> (should be 64)
            pred = pred.reshape(N, T, L, O)
            # print("Reshaped pred:", pred.shape)  # (N, T, L, P^2 * C)

            recons_per_timestep = []
            for t in range(T):
                recon_t = model.unpatchify(pred[:, t])  # (N, C, H, W)
                recons_per_timestep.append(recon_t.cpu())

            # Stack reconstructed frames back into (N, C, T, H, W)
            recons_stacked = torch.stack(recons_per_timestep, dim=1)  # (N, T, C, H, W)
            imgs_list.append(imgs.cpu())  # (N, T, C, H, W)
            recon_list.append(recons_stacked)  # (N, T, C, H, W)

            if len(imgs_list) * imgs.shape[0] >= num_images:
                break

    imgs = torch.cat(imgs_list, dim=0)[:num_images]  # (num_images, C, T, H, W)
    recons = torch.cat(recon_list, dim=0)[:num_images]  # (num_images, C, T, H, W)

    fig, axes = plt.subplots(num_images, T * 2, figsize=(8 * T, 2 * num_images))
    for i in range(num_images):
        for t in range(T):
            
            img = normalize_for_display(imgs[i, t])
            recon = (recons[i, t])

            axes[i, t * 2].imshow(img.permute(1, 2, 0).numpy())        # Original at t
            axes[i, t * 2 + 1].imshow(recon.permute(1, 2, 0).numpy())  # Reconstruction at t
            axes[i, t * 2].axis('off')
            axes[i, t * 2 + 1].axis('off')
            axes[i, t * 2].set_title(f"Original T{t}")
            axes[i, t * 2 + 1].set_title(f"Reconstruction T{t}")

    plt.show()

