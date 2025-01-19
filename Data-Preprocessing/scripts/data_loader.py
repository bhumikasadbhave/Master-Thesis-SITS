import numpy as np
import pickle
import os
from PIL import Image
import random
import shutil

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


def load_sentinel_images(sentinel_image_path, sentinel_mask_path, sentinel_id_path):
    """ Function to load all raw Sentinel-2 Images and their corresponding Sugar-beet masks
        Input: Path to the directories containing images and masks respectively
        Output: Array of Sentinel Images and the Sugar-beet Masks as numpy arrays
    """
    sentinel_images = []
    sugarbeet_masks = []
    id_masks = []

    image_files = [f for f in os.listdir(sentinel_image_path) if f.endswith('.pkl')]   #load all pkl images
    mask_files = [f for f in os.listdir(sentinel_mask_path) if f.endswith('.pkl')]     #load all pkl masks
    id_files = [f for f in os.listdir(sentinel_id_path) if f.endswith('.pkl')]

    image_files.sort()    #sort to ensure correct pairings
    mask_files.sort()
    id_files.sort()

    if len(image_files) != len(mask_files) or len(image_files)!=len(id_files):
        raise ValueError("ERROR! The number of image files does not match the number of mask files!")

    for image_file, mask_file, id_file in zip(image_files, mask_files, id_files):

        image_filepath = os.path.join(sentinel_image_path, image_file)
        mask_filepath = os.path.join(sentinel_mask_path, mask_file)
        id_filepath = os.path.join(sentinel_id_path, id_file)

        #Read image and mask
        with open(image_filepath, 'rb') as f:
            image = pickle.load(f)
        
        with open(mask_filepath, 'rb') as f:
            mask = pickle.load(f)

        with open(id_filepath, 'rb') as f:
            id = pickle.load(f)
        
        #Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        id_np = np.array(id)

        sentinel_images.append(image_np)
        sugarbeet_masks.append(mask_np)
        id_masks.append(id_np)

    return sentinel_images, sugarbeet_masks, id_masks


def save_field_images(fields_base_directory, extracted_images, train_test_ratio):
    """ Function to save save field Images. Images go to Train and Test folders based on train_test_ratio
        Input: Array of array - patches for all the given images
        Output: Boolean value indicating success/failure in the saving of images
    """
    success = False

    #Create Directories
    train_directory = os.path.join(fields_base_directory, 'train')
    test_directory = os.path.join(fields_base_directory, 'test')

    recreate_directory(train_directory)
    recreate_directory(test_directory)

    #Split Image Array
    if len(extracted_images) == 1:            #Only one image: Goes to Train (edge case)
        train_images = [extracted_images[0]]                 
        test_images = []

    elif len(extracted_images) == 2:          #Two images: One goes to Train, one to Test (edge case)
        train_images = [extracted_images[0]]  
        test_images = [extracted_images[1]]
    
    else:
        split_index = int(len(extracted_images) * train_test_ratio)    #Images go to Train and Test based on train_test_ratio
        train_images = extracted_images[:split_index]
        test_images = extracted_images[split_index:]

    #Save Patches
    for i, image_patches in enumerate(train_images):
        success = save_patches(image_patches, train_directory, i, 'train')

    for i, image_patches in enumerate(test_images):
        success = save_patches(image_patches, test_directory, i, 'test')

    return success


##### Helper Functions #####

def save_patches(image_patches, target_directory, image_index, prefix):
    """ Helper function to save extracted patches of a given image
        Input: Array of Image patches, Target Directory, Index of the current image, and prefix (test/train)
        Output: Boolean indicating success if no error occurs
    """
    for patch_index, patch in enumerate(image_patches):

        filepath = os.path.join(target_directory, f"{prefix}_image{image_index+1}_patch{patch_index+1}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(patch, f)

    return True


def recreate_directory(directory):
    """ Helper function to overwrite directory or create new if not already present 
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  
    os.makedirs(directory)


def load_test_images(path):
    pass