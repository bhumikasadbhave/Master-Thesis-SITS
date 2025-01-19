import numpy as np
import os
import pickle


def load_single_image(image_path):
    """ Load single image from given file path
        Input: Image path
        Output: Image loaded as a numpy array
    """
    with open(image_path, 'rb') as f:
        image = pickle.load(f)

    #convert to numpy array    
    image_np = np.array(image)
    return image_np

def load_sentinel_images_temporal(sentinel_base_path):
    """ Function to load all raw Sentinel-2 Images and their corresponding Sugar-beet masks.
        Input: Path to the base directory containing images and masks.
        Output: Array of Sentinel Images and the Sugar-beet Masks as numpy arrays.
    """
    sentinel_images = []
    sugarbeet_masks = []
    id_masks = []
    temporal_image_names = []

    # Get all subdirectories inside the base directory (image folders)
    image_folders = [f for f in os.listdir(sentinel_base_path) if os.path.isdir(os.path.join(sentinel_base_path, f))]
    
    for folder in image_folders:
        folder_path = os.path.join(sentinel_base_path, folder)

        # Load the corresponding mask and id mask
        mask_filepath = os.path.join(folder_path, 'fieldmask.pkl')
        id_filepath = os.path.join(folder_path, 'field_number.pkl')
        
        with open(mask_filepath, 'rb') as f:
            mask = pickle.load(f)
            mask_np = np.array(mask)

        with open(id_filepath, 'rb') as f:
            id = pickle.load(f)
            id_np = np.array(id)
        
        # Load the temporal images
        temporal_image_folder = os.path.join(folder_path, 'temporal_data')
        temporal_stack = get_temporal_stack(temporal_image_folder, mask_np, id_np)
        
        # Append the temporal stack and masks
        sentinel_images.append(temporal_stack)
        print(temporal_stack[0].shape)

    return sentinel_images 


#Helper function
def get_temporal_stack(temporal_image_folder, sugarbeet_mask, id_mask):
    """
    Load a temporal stack of images from a directory and capture their names.
    Input: Path to the folder containing temporal images.
    Output: List of numpy arrays (images in the temporal stack), AND List of image names (strings with acquisition dates or filenames).
    """
    temporal_stack = []
    #image_names = []

    image_files = [f for f in os.listdir(temporal_image_folder) if f.endswith('.pkl')]
    image_files.sort()  #Sort to maintain temporal order

    for image_file in image_files:

        image_filepath = os.path.join(temporal_image_folder, image_file)
        image = load_single_image(image_filepath)
        image_np =  np.array(image)
        image_tnp = np.transpose(image_np, (1, 2, 0))     #(Height, Width, Channels)

        ######### adding masks
        sugarbeet_mask_expanded = np.expand_dims(sugarbeet_mask, axis=-1)
        #id_mask = np.round(id_mask / 1e7, 8)                                     #from 1230818. to 0.1230818
        id_mask_expanded = np.expand_dims(id_mask, axis=-1)                       #1230818.

        #add date as another channel
        date_mask = get_date_mask(image_file, sugarbeet_mask)
        date_mask_expanded = np.expand_dims(date_mask, axis=-1)

        #combined_image = np.concatenate((image_tnp, sugarbeet_mask_expanded, id_mask_expanded, date_mask_expanded), axis=-1)
        combined_image = np.concatenate((image_tnp, id_mask_expanded, date_mask_expanded), axis=-1)
        
        if combined_image.shape[-1] != 13: 
            raise ValueError('Error in adding Sugar-Beet, Date Mask and ID Mask to the image stack!')

        temporal_stack.append(combined_image)

    return temporal_stack 


#Helper funtion
def get_date_mask(date, sugarbeet_mask):
    """Creates a Date Mask of dimension same as sugarbeet_mask. 
       The pixels have value 0.yyyymmdd where sugarbeet_mask > 0 else 0
    """
    year, month, day,_,_,_ = date.split("_")
    date_value = float(f"{year}{int(month):02d}{int(day):02d}.0")

    # Create the date mask
    date_mask = np.zeros_like(sugarbeet_mask, dtype=float)
    date_mask[sugarbeet_mask > 0] = date_value

    return np.array(date_mask)



def save_field_images_temporal(base_directory, temporal_images):
    """ Function to save temporal field patches
        Input: temporal_stack_patches - List of temporal patches for all the given images
        Output: Boolean value indicating success/failure in the saving of images
    """
    try:
        for field_idx, field_patches in enumerate(temporal_images):
            for t, temporal_image in enumerate(field_patches):

                id_mask = temporal_image[..., 11]                   # field_id
                unique_field_ids = np.unique(id_mask)
                unique_field_ids = unique_field_ids[unique_field_ids != 0]  

                if len(unique_field_ids) > 1:
                    combined_field_id = '_'.join(map(str, sorted(unique_field_ids)))
                elif len(unique_field_ids) == 1:
                    combined_field_id = str(unique_field_ids[0])
                else:
                    combined_field_id = f'field{field_idx}'         # Fallback if no valid IDs

                # Directory for every field ID
                field_folder = os.path.join(base_directory, f'{combined_field_id}')
                os.makedirs(field_folder, exist_ok=True)

                # Save temporal images
                patch_filepath = os.path.join(field_folder, f'{combined_field_id}_t{t + 1}.pkl')
                with open(patch_filepath, 'wb') as f:
                    pickle.dump(temporal_image, f)

        return True

    except Exception as e:
        print(f"Error occurred while saving field images: {e}")
        return False


def load_field_images_temporal(base_directory):
    """ Function to load temporal field patches into a list of lists
        Input: base_directory - Directory where the patches are stored
        Output: List of lists containing the loaded temporal field patches
    """
    temporal_images = []

    try:
        for field_folder in sorted(os.listdir(base_directory)):
            full_field_path = os.path.join(base_directory, field_folder)
            if os.path.isdir(full_field_path):
                temporal_patches = []
                for patch_file in sorted(os.listdir(full_field_path)):
                    patch_path = os.path.join(full_field_path, patch_file)
                    with open(patch_path, 'rb') as f:
                        temporal_patch = pickle.load(f)
                        temporal_patches.append(np.array(temporal_patch))
                temporal_images.append(temporal_patches)
        return temporal_images

    except Exception as e:
        print(f"Error occurred while loading field images: {e}")
        return []
