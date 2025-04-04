from ast import List, Tuple
from ctypes import Union
import math
from typing import Optional
import warnings
import matplotlib.pyplot as plt
import torch


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
    plt.grid(True)
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


def visualize_temporal_reconstructions(model, dataloader, device, num_images=5, T=3):
    """ Visualizes reconstructed images over multiple temporal frames.
    """
    model.eval()
    imgs_list, recon_list = [], []

    with torch.no_grad():
        for imgs, fn, timestamps in dataloader:
            imgs, timestamps = imgs.to(device), timestamps.to(device)
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


def normalize_for_display(image):
    """ Adjusts brightness & contrast for visualization. """
    image = image - image.min()  # Shift to [0, max]
    image = image / (image.max() - image.min())  # Normalize to [0,1]
    return image


def plot_reconstructed_images(model, dataloader, num_images=5, device='mps'):
    """Plots original and reconstructed images side by side"""
    
    model.eval()
    with torch.no_grad():
        for inputs_cpu, _ in dataloader:
            inputs = inputs_cpu.to(device)
            _, reconstructed = model(inputs)

            # Move tensors to CPU for visualization
            inputs = inputs.cpu()
            reconstructed = reconstructed.cpu()
            
            fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))
            
            for i in range(num_images):
                # Extract the first 3 channels from the first temporal image
                original_img = inputs[i, :3, 0, :, :].permute(1, 2, 0)  # (H, W, 3)
                original_img = normalize_for_display(original_img)
                reconstructed_img = reconstructed[i, :3, 0, :, :].permute(1, 2, 0)  # (H, W, 3)
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
