import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def integrate_sugarbeet_mask(images, sugarbeet_masks, id_masks):
    """ Add Sugarbeet Mask as 12th channel in the Sentinel Image
    """
    new_images = []
    for i in range(0,len(images)):

        combined_image = np.concatenate((images[i], sugarbeet_masks[i], id_masks[i]), axis=2)

        if combined_image.shape[2] != 13:
            raise ValueError('Error in adding Sugar-Beet Mask to the images!')

        new_images.append(combined_image)
   
    return new_images 


def mask_images(images):
    """ Mask pixels that are not sugar-beet fields and pixels that are covered by clouds
        Input: Array of images to be masked
        Output: Array of masked images
    """
    for i in range(len(images)):
        count = 0
        for x in range(0,images[i].shape[0]):
            for y in range(0,images[i].shape[1]):

                #discard pixels that don't belong to sugar-beet fields and pixels that are covered by clouds
                if images[i][x][y][10] == 0 or images[i][x][y][11] == 0:
                    images[i][x][y][:-2] = 0
                    count+=1
        print(f'--- {count} pixels masked in image {i}')
    return images


def extract_fields(images, patch_size):
    """ Function to extract single fields from an image
        Input: Array of images for which fields are to be extracted
        Output: Array of array - patches for all the given images
    """
    image_patches = []    #array of array - patches of all images

    for i in range(len(images)):
        image = images[i]  #current image

        #Label connected components in the sugarbeet field mask (12th channel)
        sugarbeet_mask = image[..., 11]       
        labeled_mask = label(sugarbeet_mask)      #Unique integer label for every connected component

        #Bounding boxes for each labeled field
        regions = regionprops(labeled_mask)

        patches = []
        for region in regions:
            
            min_row, min_col, max_row, max_col = region.bbox         #Bounding Box
            patch = image[min_row:max_row, min_col:max_col, :]       #Extract patch

            #Padding to make all patches - (patch_size x patch_size)
            height, width, channels = patch.shape
            pad_height = max(0, patch_size[0] - height)
            pad_width = max(0, patch_size[1] - width)

            #Padding amounts for top/bottom and left/right - centered padding
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # Apply padding to ensure the patch has the desired spatial size
            padded_patch = np.pad(
                patch, 
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  #Padding for height and width only
                mode='constant', 
                constant_values=0
            )

            #Crop if the patch is larger than the target size (in case a patch exceeds 50x50)
            cropped_patch = padded_patch[:patch_size[0], :patch_size[1], :]

            patches.append(cropped_patch)
        
        print(f'--- {len(patches)} sugar-beet fields extracted for image {i}')
        image_patches.append(patches)

    return image_patches


def images_vegetation_indices(images):
    """ Appends Vegetation Indices to images -- keeps other channels as well
        IP Image dimension: 1100 x 1100 x 13
        OP Image dimension: 1100 x 1100 x 17 = original + NDVI + ARI1 + ARI2 + mCAI
    """
    images_with_vegetation_indices = []
    patches = []

    for i in range(len(images)):

        for j in range(len(images[i])):

            image = images[i][j]

            blue = image[:, :, 0]        # Band 2
            green = image[:, :, 1]       # Band 3
            red = image[:, :, 2]         # Band 4
            red_edge1 = image[:, :, 3]   # Band 5
            red_edge2 = image[:, :, 4]   # Band 6
            red_edge3 = image[:, :, 5]   # Band 7
            nir = image[:, :, 6]         # Band 8
            nir_narrow = image[:, :, 7]  # Band 8A
            swir1 = image[:, :, 8]       # Band 11
            swir2 = image[:, :, 9]       # Band 12

            # Mask invalid (black) pixels
            valid_pixels = np.any(image[:, :, :10] > 0, axis=2)  
            invalid_mask = ~valid_pixels  
            np.seterr(divide='ignore', invalid='ignore')  
            
            # NDVI
            ndvi = (nir - red) / (nir + red)
            ndvi[invalid_mask] = np.nan
            ndvi = np.clip(ndvi, -1, 1)
            ndvi = ndvi[..., np.newaxis]

            # ARI1
            ari1 = (1 / green) - (1 / red_edge1)
            ari1[invalid_mask] = np.nan
            ari1 = np.clip(ari1, -1, 1)
            ari1 = ari1[..., np.newaxis]

            # ARI2
            ari2 = (nir / green) - (nir / red_edge1)
            ari2[invalid_mask] = np.nan
            ari2 = np.clip(ari2, -1, 1)
            ari2 = ari2[..., np.newaxis]

            # mCAI
            mcai = np.mean(1 - image[:, :, 4:6], axis=2)  # Bands 6–7 (Red Edge 2–3)
            mcai[invalid_mask] = np.nan
            mcai = np.clip(mcai, -1, 1)
            mcai = mcai[..., np.newaxis]

            combined_image = np.concatenate((image, ndvi, ari1, ari2, mcai), axis=2)
            patches.append(combined_image)

        images_with_vegetation_indices.append(patches)

    return images_with_vegetation_indices


def images_vegetation_indices_only(images):
    """ Appends Vegetation Indices to images -- removes non-essential channels, keeps only RGB images and their vegetation indices
    """
    required_cannels = [0,1,2,12]
    images_with_vegetation_indices = []
    patches = []

    for i in range(len(images)):

        for j in range(len(images[i])):

            image = images[i][j]

            blue = image[:, :, 0]        # Band 2
            green = image[:, :, 1]       # Band 3
            red = image[:, :, 2]         # Band 4
            red_edge1 = image[:, :, 3]   # Band 5
            red_edge2 = image[:, :, 4]   # Band 6
            red_edge3 = image[:, :, 5]   # Band 7
            nir = image[:, :, 6]         # Band 8
            nir_narrow = image[:, :, 7]  # Band 8A
            swir1 = image[:, :, 8]       # Band 11
            swir2 = image[:, :, 9]       # Band 12

            # Mask invalid (black) pixels
            valid_pixels = np.any(image[:, :, :10] > 0, axis=2)  
            invalid_mask = ~valid_pixels  
            np.seterr(divide='ignore', invalid='ignore')  
            
            # NDVI
            ndvi = (nir - red) / (nir + red)
            ndvi[invalid_mask] = np.nan
            ndvi = np.clip(ndvi, -1, 1)
            ndvi = ndvi[..., np.newaxis]

            # ARI1
            ari1 = (1 / green) - (1 / red_edge1)
            ari1[invalid_mask] = np.nan
            ari1 = np.clip(ari1, -1, 1)
            ari1 = ari1[..., np.newaxis]

            # ARI2
            ari2 = (nir / green) - (nir / red_edge1)
            ari2[invalid_mask] = np.nan
            ari2 = np.clip(ari2, -1, 1)
            ari2 = ari2[..., np.newaxis]

            # mCAI
            mcai = np.mean(1 - image[:, :, 4:6], axis=2)  # Bands 6–7 (Red Edge 2–3)
            mcai[invalid_mask] = np.nan
            mcai = np.clip(mcai, -1, 1)
            mcai = mcai[..., np.newaxis]

            # Remove non-essential channels
            filtered_image = remove_channels(image, required_cannels)

            combined_image = np.concatenate((filtered_image, ndvi, ari1, ari2, mcai), axis=2)
            patches.append(combined_image)
        
        images_with_vegetation_indices.append(patches)

    return images_with_vegetation_indices


def remove_channels(image, keep_indices):
    """
    Removes specified channels from a Sentinel-2 image.
    """
    # Select only the given channels
    filtered_image = image[:, :, keep_indices]
    return filtered_image

#temp
def get_average_indices(images):
    """
    Calculate NDVI, ARI1, ARI2, and mCAI for all patches in all images.
    """
    results_ndvi = []
    results_ari1 = []
    results_ari2 = []
    results_mcai = []

    for i in range(len(images)):

        ndvi_patches = []
        ari1_patches = []
        ari2_patches = []
        mcai_patches = []

        for j in range(len(images[i])):

            image = images[i][j]

            nir = image[:, :, 6]    # NIR -> Band 8
            red = image[:, :, 2]    # Red -> Band 4
            r550 = image[:, :, 1]   # Band at 550 nm (Green)
            r700 = image[:, :, 3]   # Band at 700 nm (Red edge)
            r800 = image[:, :, 6]   # Band at 800 nm (Near-Infrared)

            # Avoid division errors
            np.seterr(divide='ignore', invalid='ignore')

            # NDVI
            ndvi = (nir - red) / (nir + red)
            ndvi = np.clip(ndvi, -1, 1)
            ndvi_patches.append(np.nanmean(ndvi))

            # ARI1
            ari1 = (1 / r550) - (1 / r700)
            ari1 = np.clip(ari1, -1, 1)
            ari1_patches.append(np.nanmean(ari1))

            # ARI2
            ari2 = (r800 / r550) - (r800 / r700)
            ari2 = np.clip(ari2, -1, 1)
            ari2_patches.append(np.nanmean(ari2))

            # mCAI
            image[:, :, 4:6][image[:, :, 4:6] == 0] = np.nan
            mcai = np.nanmean(1 - image[:, :, 4:6], axis=2)
            mcai = np.clip(mcai, -1, 1)
            mcai_patches.append(np.nanmean(mcai))

        # Append average patch results for the entire image
        results_ndvi.append(ndvi_patches)
        results_ari1.append(ari1_patches)
        results_ari2.append(ari2_patches)
        results_mcai.append(mcai_patches)

    return results_ndvi, results_ari1, results_ari2, results_mcai

#temp
def calculate_indices(image):
    """
    Calculate multiple vegetation indices for a given image.
    """

    indices = {}

    blue = image[:, :, 0]    # Band 2
    green = image[:, :, 1]   # Band 3
    red = image[:, :, 2]     # Band 4
    red_edge1 = image[:, :, 3]  # Band 5
    red_edge2 = image[:, :, 4]  # Band 6
    red_edge3 = image[:, :, 5]  # Band 7
    nir = image[:, :, 6]     # Band 8
    nir_narrow = image[:, :, 7]  # Band 8A
    swir1 = image[:, :, 8]   # Band 11
    swir2 = image[:, :, 9]   # Band 12

    # Mask invalid (black) pixels
    valid_pixels = np.any(image[:, :, :10] > 0, axis=2)  # Valid if any band is >0
    invalid_mask = ~valid_pixels  # Black or invalid pixels

    np.seterr(divide='ignore', invalid='ignore')  # Ignore divide warnings

    # NDVI
    ndvi = (nir - red) / (nir + red)
    ndvi[invalid_mask] = np.nan
    indices['NDVI'] = ndvi

    # ARI1
    ari1 = (1 / green) - (1 / red_edge1)
    ari1[invalid_mask] = np.nan
    indices['ARI1'] = ari1

    # ARI2
    ari2 = (nir / green) - (nir / red_edge1)
    ari2[invalid_mask] = np.nan
    indices['ARI2'] = ari2

    # SAVI
    L = 0.5
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    savi[invalid_mask] = np.nan
    indices['SAVI'] = savi

    # EVI
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    evi[invalid_mask] = np.nan
    indices['EVI'] = evi

    # mCAI
    mcai = np.mean(1 - image[:, :, 4:6], axis=2)  # Bands 6–7 (Red Edge 2–3)
    mcai[invalid_mask] = np.nan
    indices['mCAI'] = mcai

    # GNDVI
    gndvi = (nir - green) / (nir + green)
    gndvi[invalid_mask] = np.nan
    indices['GNDVI'] = gndvi

    # RDVI
    rdvi = (nir - red) / np.sqrt(nir + red)
    rdvi[invalid_mask] = np.nan
    indices['RDVI'] = rdvi

    # CIgreen
    cigreen = (nir / green) - 1
    cigreen[invalid_mask] = np.nan
    indices['CIgreen'] = cigreen

    # MSAVI
    msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
    msavi[invalid_mask] = np.nan
    indices['MSAVI'] = msavi

    # Clip all indices
    for key in indices:
        indices[key] = np.clip(indices[key], -1, 1)

    return indices
