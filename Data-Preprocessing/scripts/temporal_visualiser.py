import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import torch

######## Visualising Temporal Cubes - Based on Index ########


# RGB
def visualize_temporal_stack_rgb(temporal_stack, dates=None, num_timesteps=7):
    """
    Visualize all images in a single temporal stack with RGB channels and acquisition dates as titles.
    """

    if isinstance(temporal_stack, torch.Tensor):
        temporal_stack = temporal_stack.detach().cpu().numpy() 
        temporal_stack = np.transpose(temporal_stack, (0, 2, 3, 1))   # T, H, W, C

    fig, axes = plt.subplots(1, num_timesteps, figsize=(20, 5))
    fig.suptitle("Patch-level Temporal Stack Visualization (RGB)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        # date = dates[i]
        
        # if len(date) > 0:
        #     int_date = int(float(date))  # yyyymmdd.0
        #     year = int_date // 10000
        #     month = (int_date // 100) % 100
        #     day = int_date % 100
        #     acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        # else:
        #     acquisition_date = "No Date"
        
        print(image.shape)
        rgb_image = np.stack([image[..., 2], image[..., 1], image[..., 0]], axis=-1)    # RGB: BGR -> RGB
        ax.imshow(rgb_image)
        ax.imshow(np.clip(rgb_image / np.max(rgb_image), 0, 1), cmap='viridis')         # Normalize for display
        ax.set_title('acquisition_date', fontsize=10)
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the subtitle
    plt.show()


# VI Differences
def visualize_ndvi_difference(temporal_stack, num_timesteps=6):
    """
    Visualize NDVI difference images in a temporal stack with acquisition dates as titles.
    Args:
        temporal_stack (list): A list of NDVI difference images (H, W) for each timestep.
        dates (list): A list of dates corresponding to each image in temporal_stack.
        num_timesteps (int): The number of timesteps to visualize (default is 6).
    """
    fig, axes = plt.subplots(1, num_timesteps, figsize=(20, 5))
    fig.suptitle("NDVI Difference Visualization (Temporal Stack)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        # date = dates[i]
        
        # if len(date) > 0:
        #     int_date = int(float(date))  # yyyymmdd.0
        #     year = int_date // 10000
        #     month = (int_date // 100) % 100
        #     day = int_date % 100
        #     acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        # else:
        #     acquisition_date = "No Date"
        
        im = ax.imshow(image)  # Use a colormap that works for NDVI-like data
        ax.set_title('ndvi diff', fontsize=10)
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the subtitle
    plt.show()



# NDVI
def visualize_temporal_stack_ndvi(temporal_stack, dates):
    """
    Visualize NDVI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (NDVI)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        # Calculate NDVI
        nir = image[..., 6]
        red = image[..., 2]
        # ndvi = (nir - red) / (nir + red + 1e-6)         # Avoid division by zero

        np.seterr(divide='ignore', invalid='ignore')  #Avoid division errors
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1) 
        
        ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])

        # ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ARI
def visualize_temporal_stack_ari(temporal_stack, dates):
    """
    Visualize ARI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (ARI)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        # Calculate ARI
        green = image[..., 1]
        red_edge1 = image[..., 3]

        np.seterr(divide='ignore', invalid='ignore')  #Avoid division errors
        ari = (1 / (green)) - (1 / (red_edge1))
        ari = np.clip(ari, -1, 1)
        
        ax.imshow(ari, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# mCAI
def visualize_temporal_stack_mcai(stack, dates):
    """
    Visualize mCAI for a temporal stack of Sentinel-2 images.
    Args: stack (list of np.array): Temporal stack of Sentinel-2 images. Each image is expected to have 14 channels.
    """
    
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (mCAI)", fontsize=16)
    
    for i, image in enumerate(stack):
        
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"

        # Compute mCAI
        valid_pixels = np.any(image[:, :, :11] > 0, axis=2)     # Valid if any band is >0
        invalid_mask = ~valid_pixels                            # Black or invalid pixels
        mcai = np.mean(1 - image[:, :, 4:6], axis=2)            # Bands 6–7 (Red Edge 2–3)
        mcai[invalid_mask] = np.nan
        mcai = np.clip(mcai, 0, 1)

        # Plot mCAI
        plt.subplot(1, len(stack), i + 1)
        plt.imshow(mcai, cmap='viridis', vmin=0, vmax=1)
        plt.title(f"mCAI\n{acquisition_date}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# CI
def visualize_temporal_stack_ci(temporal_stack, dates):
    """
    Visualize CI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (CI)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        # Calculate CI
        nir = image[..., 6]       #  Band 8 (NIR) 
        red_edge = image[..., 3]  #  Band 5 (Red Edge) 

        np.seterr(divide='ignore', invalid='ignore')  # Avoid division errors
        ci = (nir - red_edge) / (nir + red_edge)
        ci = np.clip(ci, 0, 1)  
        
        ax.imshow(ci, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# MCARI
def visualize_temporal_stack_mcari(temporal_stack, dates):
    """
    Visualize MCARI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (MCARI)", fontsize=16)
    
    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        # Calculate MCARI
        green = image[..., 1]      # Band 3 (Green) 
        red = image[..., 2]        # Band 4 (Red) 
        red_edge1 = image[..., 3]  # Band 5 (Red Edge 1) 

        np.seterr(divide='ignore', invalid='ignore')                        # Avoid division errors
        mcari = ((red_edge1 - red) - 0.2 * (red_edge1 - green) * (red_edge1 / red))
        mcari = np.clip(mcari, 0, 1)                 
        
        ax.imshow(mcari, cmap='viridis', vmin=0, vmax=np.percentile(mcari, 95))
        ax.set_title(acquisition_date, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# PRI
def visualize_temporal_stack_pri(temporal_stack, dates):
    """
    Visualize PRI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (PRI)", fontsize=16)

    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        green = image[..., 1] 
        red_edge1 = image[..., 3] 

        np.seterr(divide='ignore', invalid='ignore')
        pri = (green - red_edge1) / (green + red_edge1)
        pri = np.clip(pri, -1, 1)

        ax.imshow(pri, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

#NDRE
def visualize_temporal_stack_ndre(temporal_stack, dates):
    """
    Visualize NDRE for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (NDRE)", fontsize=16)

    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        nir = image[..., 6] 
        red_edge1 = image[..., 3] 

        np.seterr(divide='ignore', invalid='ignore')
        ndre = (nir - red_edge1) / (nir + red_edge1)
        ndre = np.clip(ndre, -1, 1)

        ax.imshow(ndre, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# EVI
def visualize_temporal_stack_evi(temporal_stack, dates):
    """ Visualize EVI for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """

    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (EVI)", fontsize=16)

    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        nir = image[..., 6] 
        red = image[..., 2] 
        blue = image[..., 0] 

        np.seterr(divide='ignore', invalid='ignore')
        valid_pixels = np.any(image[:, :, :11] > 0, axis=2)     # Valid if any band is >0
        invalid_mask = ~valid_pixels 
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        evi[invalid_mask] = np.nan
        evi = np.clip(evi, -1, 1)

        ax.imshow(evi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(acquisition_date, fontsize=10)
        # ax.axis("off")
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    
    plt.tight_layout()
    plt.show()

# MSI
def visualize_temporal_stack_msi(temporal_stack, dates):
    """
    Visualize MSI (Moisture Stress Index) for all images in a single temporal stack with acquisition dates as titles.
    Args: temporal_stack (list): A single temporal stack (list of 7 images).
    """
    fig, axes = plt.subplots(1, 7, figsize=(20, 5))
    fig.suptitle("Temporal Stack Visualization (MSI)", fontsize=16)

    for i, ax in enumerate(axes):
        image = temporal_stack[i]
        date = dates[i]
        
        # Acquisition date
        if len(date) > 0:
            int_date = int(float(date)) # yyyymmdd.0
            year = int_date // 10000
            month = (int_date // 100) % 100
            day = int_date % 100
            acquisition_date = datetime(year, month, day).strftime("%Y-%m-%d")
        else:
            acquisition_date = "No Date"
        
        # Calculate MSI
        nir = image[..., 6]        # NIR (Band 8)
        swir1 = image[..., 8]      # SWIR1 (Band 11)

        np.seterr(divide='ignore', invalid='ignore')  # Avoid division errors
        msi = swir1 / nir 
        msi = np.clip(msi, 0, 2)  
        
        ax.imshow(msi, cmap='coolwarm', vmin=0, vmax=2)
        ax.set_title(acquisition_date, fontsize=10)
        # ax.axis("off")
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])

    plt.tight_layout()
    plt.show()


######## Helper Functions ########

# Helper function to test how VI differences look like
def visualize_temporal_differences(temporal_differences):
    """
    Visualizes the temporal differences for a single field in a single row,
    with NaN and 0 values displayed as white, and green for high changes, red for low.
    """
    if len(temporal_differences) == 0:
        print("No temporal differences to visualize.")
        return

    num_images = len(temporal_differences)
    
    # Set figure size dynamically based on the number of images
    plt.figure(figsize=(5 * num_images, 5))  
    
    # Determine consistent color range for all images (excluding NaNs and 0s)
    vmin = np.nanmin([np.nanmin(diff) for diff in temporal_differences])
    vmax = np.nanmax([np.nanmax(diff) for diff in temporal_differences])
    
    # Use an existing colormap (coolwarm) and reverse it
    cmap = plt.cm.coolwarm.reversed() 
    
    # Ensure NaNs are white
    cmap.set_bad(color='white')  # NaN values

    for i, diff in enumerate(temporal_differences):
        # Mask both 0 and NaN values to ensure background pixels stay white
        masked_diff = np.ma.masked_equal(diff, 0)  # Mask 0 values
        masked_diff = np.ma.masked_invalid(masked_diff)  # Mask NaN values

        plt.subplot(1, num_images, i + 1)
        # Display only non-zero values using masked array
        plt.imshow(masked_diff, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        plt.title(f"Temporal Diff {i + 1}", fontsize=10)
        plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Helper Function for creating plots for the Manuscript
def visualise_selected_bands(spectral_image):
    """ Visualise specific Sentinel bands: 0 (Blue), 2 (Red), 6 (NIR), 8 (SWIR1) """

    selected_bands = [0, 2, 6, 8]
    band_labels = ["Blue", "Red", "NIR", "SWIR1"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Single row, 4 columns
    
    for idx, (band, label) in enumerate(zip(selected_bands, band_labels)):
        ax = axes[idx]
        im = ax.imshow(spectral_image[:, :, band], cmap='viridis')  
        
        ax.set_title(f"{label} (Band {band+2})", fontsize=20)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60]) 
        ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.show()