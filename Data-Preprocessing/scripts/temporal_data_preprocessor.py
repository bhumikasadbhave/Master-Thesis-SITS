import numpy as np
from datetime import datetime
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion
import pandas as pd


def mask_images_temporal(images):
    """ Mask pixels that are not sugar-beet fields
        Input: Array of temporal stacks of images with sugar-beet mask and id mask integrated as channels
        Output: Array of masked temporal stacks
    """
    for img_index in range(len(images)):
        temporal_stack = images[img_index]  # Get the temporal stack (list of 3D images)
        count = 0

        for temporal_index in range(len(temporal_stack)):
            image = temporal_stack[temporal_index]  

            for x in range(image.shape[0]):
                for y in range(image.shape[1]):

                    if image[x, y, 11] == 0:
                        image[x, y, :-2] = 0
                        count += 1

            # print(f'--- {count} pixels masked in temporal instance {temporal_index} of image {img_index}')
    return images


def extract_fields_temporal(images, patch_size):
    """
    Extract sugar-beet field patches from temporal images, pad them to fixed size.
    Args: images (list): A list of scenes, where each scene is a list of T temporal images, each image of size HxWxC.
          patch_size (tuple): Desired patch size as (height, width).
    Returns: list: A list of patches, where each patch is a list of T temporal images, all padded to `patch_size`.
    """
    all_patches = []  

    for scene_index, temporal_images in enumerate(images):
        T = len(temporal_images) 
        
        sugarbeet_id_mask = temporal_images[0][..., 11]             # Contains ID where there is a sugarbeet field, else 0
        binary_mask = (sugarbeet_id_mask > 0).astype(np.uint8)      # Binarize the mask before label components step
        labeled_mask = label(binary_mask)                           # Label unique components in the mask
        regions = regionprops(labeled_mask)                         # Extract properties of labeled regions

        for region_index, region in enumerate(regions):

            min_row, min_col, max_row, max_col = region.bbox        # Bounding box dimensions for each region
            
            # Collect and pad patches for the current region across all timepoints
            temporal_patch = []
            for img in temporal_images:
                
                # Extract raw patch using the bounding box
                patch = img[min_row:max_row, min_col:max_col, :]

                # Padding
                height, width, channels = patch.shape
                pad_height = max(0, patch_size[0] - height)
                pad_width = max(0, patch_size[1] - width)
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left

                padded_patch = np.pad(
                    patch,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),  # Padding height/width only
                    mode='constant',
                    constant_values=0
                )

                # Clip the patch if it exceeds the desired size
                padded_patch = padded_patch[:patch_size[0], :patch_size[1], :]
        
                temporal_patch.append(padded_patch)
            all_patches.append(temporal_patch)
        print(f"--- Processed {len(regions)} regions for scene {scene_index}")
    return all_patches


def refine_temporal_stack_interval5(temporal_stack_patches, stack_size, date_ranges):
    """
    Refine temporal stacks by selecting cloud-free images within specified date ranges and ensuring a 5-day gap between selected images.
    Args: temporal_stack_patches (list): List of patches, where each patch is a temporal stack of images
                                       Each image has 12 channels:
                                       - 12th channel: date values in yyyymmdd.0 format
                                       - 11th channel: sugarbeet field mask (1 indicates field, 0 no field)
                                       - 10th channel: cloud mask (1 indicates cloud, 0 no cloud)
        stack_size (int): Number of date ranges (should match len(date_ranges))
        date_ranges (list): List of tuples (label, start_date, end_date) with date ranges
    Returns: list: Refined temporal stack patches that meet the criteria
    """
    final_patches = []
    
    for patch_stack in temporal_stack_patches:
        patch_flags = np.zeros(stack_size)
        refined_patch_stack = []
        selected_dates = []  

        for idx, (_, start_date, end_date) in enumerate(date_ranges):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            date_selected = False
            images_found = 0  

            for temporal_image in patch_stack:
                date_channel = temporal_image[..., 12]          # 13th channel: date values
                field_mask = temporal_image[..., 11]            # 12th channel: sugar-beet field id mask
                binary_mask = (field_mask > 0).astype(np.uint8) # Binary mask for sugar-beet fields
                cloud_mask = temporal_image[..., 10]            # 10th channel: cloud mask
                valid_dates = date_channel[date_channel != 0]   # Non-zero date values

                if len(valid_dates) == 0:
                    continue

                int_date = int(valid_dates[0])  # Date: yyyymmdd.0
                year = int_date // 10000
                month = (int_date // 100) % 100
                day = int_date % 100
                image_date = datetime(year, month, day)

                # Check if the image is within the date range and meets the 5-day condition
                if start_date <= image_date <= end_date:
                    # Ensure a minimum 5-day gap from already selected dates
                    if all(abs((image_date - prev_date).days) >= 5 for prev_date in selected_dates):
                        field_pixels = binary_mask == 1                          # Identify pixels corresponding to sugar-beet fields
                        cloud_free = np.all(cloud_mask[field_pixels] == 1)       # Check if field pixels are cloud-free
                        if cloud_free:
                            refined_patch_stack.append(temporal_image)
                            patch_flags[idx] = 1
                            selected_dates.append(image_date)  # Store the selected date
                            images_found += 1

                            if idx == 3:  # Only one image for September
                                break

                            if images_found == 2:  # Stop after finding two valid images for this range
                                patch_flags[idx + len(date_ranges)] = 1
                                break

        # Append the patch stack only if all date ranges have at least one image
        if np.all(patch_flags != 0):
            final_patches.append(refined_patch_stack)
        else:
            print(f"Patch discarded: Missing images for some date ranges.")
            print(f"Flag array was: ", patch_flags)

    return final_patches


def blacken_field_borders_temporal(images):
    """
    Blacken the border pixels of sugarbeet fields in channels 0-9 for all temporal images based on the mask in channel 11.
    """
    modified_images = images.copy()  

    for i in range(len(images)):  
       
        sugarbeet_mask = images[i][0][:, :, 11] > 0  

        # Erode the mask to identify inner field pixels
        eroded_mask = binary_erosion(sugarbeet_mask, structure=np.ones((3, 3)))  # Shrink field region
        border_mask = sugarbeet_mask & ~eroded_mask      # Border pixels are the difference between the original and eroded masks

        for t in range(len(images[i])): 
            for channel in range(10):  # Channels 0 to 9
                modified_images[i][t][:, :, channel][border_mask] = 0
            modified_images[i][t][:, :, 11][border_mask] = 0

    return modified_images


def normalize_images(temporal_images):
    """
    Normalize the Sentinel-2 temporal images such that every channel in each temporal image is scaled to [0, 1],
    but only if it's needed (i.e., if the values are not already in the [0, 1] range).
    """
    normalized_images = []
    for field_images in temporal_images:
        field_normalized_images = []
        
        for temporal_image in field_images:
            normalized_temporal_image = np.zeros_like(temporal_image, dtype=np.float32)
            num_channels = temporal_image.shape[2]
            
            for c in range(num_channels): 

                # Skip normalization for the last 3 channels (masks etc.)
                if c >= num_channels - 3:
                    normalized_temporal_image[:, :, c] = temporal_image[:, :, c]

                else:
                    band = temporal_image[:, :, c]
                    band_min = np.min(band)
                    band_max = np.max(band)
                    
                    if band_min >= 0 and band_max <= 1:
                        # If the band is already in the [0, 1] range, don't normalize
                        normalized_band = band
                    else:
                        # Normalize the band to [0, 1] range
                        if band_max > band_min:
                            normalized_band = (band - band_min) / (band_max - band_min)
                        else:
                            normalized_band = np.zeros_like(band)  # If no variance, just set to zeros
                    normalized_temporal_image[:, :, c] = normalized_band
                    
            field_normalized_images.append(normalized_temporal_image)
        normalized_images.append(field_normalized_images)
    return normalized_images


def filter_non_sugarbeet_fields(temporal_images, sugarbeet_content_csv_path):
    """Based on the provied csv, remove the fields which are non-sugarbeet-fields"""

    sugarbeet_df = pd.read_csv(sugarbeet_content_csv_path)
    valid_fields = set(sugarbeet_df['FIELDUSNO'].astype(int).unique())
    filtered_images = []
    for img in temporal_images:
        field_numbers = np.unique(img[0][:, :, -2])
        non_zero_fields = field_numbers[field_numbers != 0]
        if any(fn in valid_fields for fn in non_zero_fields):       #keep the images with atleast 1 valid field number
            filtered_images.append(img)
    return filtered_images


# Function to get the Non-temporal instances for extracted patches. They are later used for performing preliminary test 
def get_non_temporal_images(temporal_images):
    """
    Get a single september image as Non-temporal data 
    """
    simple_images = []

    for stack in temporal_images:
        image = stack[-1]   #sept img
        simple_images.append(image)
    
    return simple_images


