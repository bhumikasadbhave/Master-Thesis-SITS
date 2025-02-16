from scripts.preprocess_helper import *
import numpy as np
import matplotlib.pyplot as plt

def visualise_rgb(spectral_image, cmap='viridis'):
    """ Funcrion to visulaise the Sentinel-2 spectral image as an RGB image
    """

    bgr_image = spectral_image[..., :3] #RGB bands 

    #Normalize the pixel values to [0-1] for every channel
    normalised_image = normalise_pixels(bgr_image)

    #Gamma correction for contrast
    gamma = 0.4  
    gamma_corrected_image = gamma_correct(normalised_image, gamma)

    #Brightness
    gain = [1.2, 1.5, 1.2]       
    brightened_image = adjust_brightness(gamma_corrected_image, gain)

    #Clip image since multiplication by gain can push the pixel values outside [0,1] range
    clipped_image = np.clip(brightened_image, 0, 1)

    #Revert image back to float 32 since it gets converted to float64 after multiplication by scalar (gain)
    clipped_image = clipped_image.astype(np.float32) 

    #Convert BGR image to RGB for plotting
    rgb_image = clipped_image[..., ::-1] 

    #Plot the image
    plot_image(rgb_image, "Sentinel-2 RGB Composite", cmap)


def visualise_single_band(spectral_image, band_index, cmap='gray'):
    """ Funcrion to visulaise a single Sentinel-2 band/channel
    """
    #Select the band
    selected_band = spectral_image[:, :, band_index]

    #Normalise the pixels
    normalised_band = normalise_pixels(selected_band)

    #Plot the image
    plot_image(normalised_band, f"Sentinel-2 Band {band_index}", cmap)


def visualise_all_bands(spectral_image):
    """ Visualise all 10 bands in a grid - masks excluded
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
            band_name = 'Sugarbeet Mask'
        else:
            band_name = f"Sentinel Band {i+2}"
            
        ax.set_title(band_name)  

    plt.tight_layout()
    plt.show()


def visualise_all_bands_wo_masks(spectral_image):
    """ Visualise the first 10 Sentinel bands in a grid - masks excluded """
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # Adjusted to 2 rows, 5 columns
    for i in range(10):  # Only iterate till i=9
        ax = axes[i // 5, i % 5]  # Adjusted to 5 images per row
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


def plot_image(image, title, cmap):
    """ Function to plot the image
    """
    plt.imshow(image, cmap)
    plt.title(title)
    plt.show()


#temp
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



#temp
def plot_index_image(average_ndvi_image):

    plt.figure(figsize=(12, 6))
    indices = range(len(average_ndvi_image))
    plt.bar(indices, average_ndvi_image, color='lightgreen', alpha=0.6, label='Average Index (Bar)')
    plt.plot(indices, average_ndvi_image, marker='o', color='green', label='Average Index (Line)', linewidth=2)
    plt.title("Average Values for all Sugarbeet fields inside the Sample Image", fontsize=16)
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("Average Vegetation Index", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.6)
    plt.show()

#temp
def visualize_indices(indices):
    """
    Visualize vegetation indices in a grid.
    """
    num_indices = len(indices)
    grid_size = int(np.ceil(np.sqrt(num_indices)))

    plt.figure(figsize=(15, 15))
    for i, (name, index) in enumerate(indices.items(), 1):
        plt.subplot(grid_size, grid_size, i)
        plt.imshow(index, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()