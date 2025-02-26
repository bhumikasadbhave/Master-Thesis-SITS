import config as config
from scripts.temporal_data_loader import *
from scripts.temporal_data_preprocessor import *
from scripts.data_visualiser import *
from scripts.preprocess_helper import *
from scripts.data_visualiser import *
from scripts.data_loader import *
from scripts.data_preprocessor import *
from scripts.temporal_data_preprocessor import *
from scripts.temporal_data_loader import *
from scripts.temporal_visualiser import *
from scripts.temporal_chanel_refinement import *
from model_scripts.get_statistics import *
from model_scripts.dataset_creation import *
from model_scripts.train_model_ae import *
from model_scripts.model_visualiser import *
from Pipeline.pre_processing_pipeline import *


class PreProcessingPipelineTemporal:

    def __init__(self):

        #Get values from Config file
        self.sentinel_base_path = config.sentinel_base_path
        self.fields_base_directory_temporal = config.base_directory_temporal
        self.field_size = config.field_size
        self.temporal_stack_size = config.temporal_stack_size
        self.date_ranges = config.temporal_points


    def run_temporal_patch_save_pipeline(self):
        """
        Function to run the patch-save preprocessing pipeline for temporal image stacks.
        It loads images, integrates masks, applies masking, extracts patches, and saves them.
        """

        # Step 1: Load Sentinel Images and Corresponding Masks
        images = load_sentinel_images_temporal(self.sentinel_base_path)
        print(f"Loaded {len(images)} temporal images and with attached masks in them.")

        # Step 2: Mask images
        masked_images = mask_images_temporal(images)
        print(f"Masked {len(masked_images)} images.")

        # Step 3: Extract patches from the masked temporal images 
        fields = extract_fields_temporal(masked_images, self.field_size)
        print(f"Extracted {len(fields)} patches from temporal images.")

        # Setp 4: Refine the temporal stack: 7 cloud-free images per patch
        refined_fields = refine_temporal_stack_interval5(fields, self.temporal_stack_size, self.date_ranges)

        # Step 5: Define the base directory to save patches
        fields_base_directory = config.base_directory_temporal

        # Step 7: Save the patches to disk in their respective temporal folders
        print("Saving patches to disk...")
        success = save_field_images_temporal(fields_base_directory, refined_fields)
        if success:
            print(f"Successfully saved the patches to {fields_base_directory}.")
        else:
            print("Failed to save the patches.")
        return success

    
    def get_processed_temporal_cubes(self, dataset_type, bands, vi_type='msi'):
        """ 
        Generalized pipeline to load the saved field patches, remove border pixels, 
        and get final model-ready temporal cube for both train and test data.
        Returns image tensor and field numbers.
        
        Parameters:
            dataset_type (str): 'train' or 'test' to specify dataset.
            batch_size (int): Batch size for loading the data.
            bands (str): Type of band selection method.
            vi_type (str, optional): Type of vegetation index in case 'indexbands' OR 'indexonly' is used, default is 'msi'.
        """

        # Step 1: Load the saved patches from the file system
        if dataset_type == 'train':
            temporal_images = load_field_images_temporal(config.base_directory_temporal_train1)
        elif dataset_type == 'test':
            temporal_images = load_field_images_temporal(config.base_directory_temporal_test1)
        else:
            raise ValueError("dataset_type must be either 'train' or 'test'")

        # Step 2: Remove the border pixels of the sugarbeet fields
        border_removed_images = blacken_field_borders_temporal(temporal_images)

        # Step 3: Select relevant Vegetation Indices and Sentinel-2 Bands
        band_selection_methods = {
            'indexbands': indexbands_temporal_cubes,
            'indexonly': indexonly_temporal_cubes,
            'relevantbands': relevantbands_temporal_cubes,
            'multipleindices': multiple_indices_temporal_cubes,
            'multipleindicesbands': multiple_indices_bands_temporal_cubes,
            'allbands': allbands_temporal_cubes
        }

        if bands not in band_selection_methods:
            raise ValueError(f"Invalid bands option: {bands}")

        if bands in ['indexbands', 'indexonly']:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](border_removed_images, vi_type)
        else:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](border_removed_images)

        # Step 4: Return Temporal Cubes
        image_tensor = np.stack(indices_images)
        
        return field_numbers, acquisition_dates, image_tensor

