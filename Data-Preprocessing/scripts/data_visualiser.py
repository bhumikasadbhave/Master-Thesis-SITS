from scripts.preprocess_helper import *
import numpy as np
import matplotlib.pyplot as plt

def visualise_rgb(spectral_image, title="Sentinel-2 RGB Composite"):
    """
    Visualize a single Sentinel-2 spectral image as an RGB image.
    """
    # Extract RGB channels (assuming BGR format)
    rgb_image = np.stack([spectral_image[..., 2], spectral_image[..., 1], spectral_image[..., 0]], axis=-1)  # Convert BGR to RGB

    # Normalize for display
    rgb_image = np.clip(rgb_image / np.max(rgb_image), 0, 1) if np.max(rgb_image) > 0 else rgb_image

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image, cmap='viridis')
    plt.title(title, fontsize=14)
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    plt.yticks([0, 10, 20, 30, 40, 50, 60])
    plt.show()


def visualise_single_band(spectral_image, band_index, cmap='gray'):
    """ Funcrion to visulaise a single Sentinel-2 band/channel
    """
    selected_band = spectral_image[:, :, band_index]
    print(selected_band)
    # normalised_band = normalise_pixels(selected_band)
    plot_image(selected_band, f"Sentinel-2 Band {band_index}",cmap)


def visualise_all_bands(spectral_image):
    """ Visualise all Sentinel-2 bands in a grid - masks included
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  
    for i in range(12):
        ax = axes[i // 4, i % 4]  
        im = ax.imshow(spectral_image[:, :, i], cmap='viridis')  

        if i==7:
            band_name = "Sentinel Band 8A"
        elif i==8:
            band_name = "Sentinel Band 11"
        elif i==9:
            band_name = "Sentinel Band 12"
        elif i==10:
            band_name = 'Cloud Mask'
        elif i==11:
            band_name = 'Sugarbeet Field Number Mask'
        else:
            band_name = f"Sentinel Band {i+2}"
            
        ax.set_title(band_name)  
    plt.tight_layout()
    plt.show()


def visualise_all_bands_wo_masks(spectral_image):
    """ Visualise the first 10 Sentinel bands in a grid - masks excluded """

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  
    for i in range(10): 
        ax = axes[i // 5, i % 5]  
        im = ax.imshow(spectral_image[:, :, i], cmap='viridis')  

        if i == 7:
            band_name = "Sentinel Band 8A"
        elif i == 8:
            band_name = "Sentinel Band 11"
        elif i == 9:
            band_name = "Sentinel Band 12"
        else:
            band_name = f"Sentinel Band {i+2}"
            
        ax.set_title(band_name)  
    plt.tight_layout()
    plt.show()


def visualize_image_channels(image):
    """
    Visualizes the channels of a given image dynamically based on the number of channels.
    """
    num_channels = image.shape[2]  
    
    cols = int(np.ceil(np.sqrt(num_channels)))  # Number of columns
    rows = int(np.ceil(num_channels / cols))   # Number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() 

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(image[:, :, i], cmap='viridis')  
        ax.set_title(f'Channel {i}', fontsize=10)
        ax.axis('off')

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


######## Helper Functions ########
def plot_image(image, title, cmap):
    """ Function to plot the image
    """
    plt.imshow(image, cmap)
    plt.title(title)
    plt.show()