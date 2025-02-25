import preprocessing_config as config
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


    def run_full_preprocessing_pipeline_temporal(self):
        """
        Function to run the complete preprocessing pipeline for temporal image stacks.
        It loads images, integrates masks, applies masking, extracts patches, and saves them.
        """

        # Step 1: Load Sentinel Images and Corresponding Masks
        images = load_sentinel_images_temporal(self.sentinel_base_path)
        print(f"Loaded {len(images)} temporal image stacks, {len(sugarbeet_masks)} sugarbeet masks, and {len(id_masks)} ID masks.")

        # Step 2: Mask images
        masked_images = mask_images_temporal(images)
        print(f"Masked {len(masked_images)} images.")

        # Step 3: Extract patches from the masked temporal images # this will be step 3
        fields = extract_fields_temporal(masked_images, self.field_size)
        print(f"Extracted {len(fields)} patches from temporal images.")

        # Setp 4: Refine the temporal stack
        refined_fields = refine_temporal_stack_raw(fields, self.temporal_stack_size, self.date_ranges)

        # Step 5: Define the base directory to save patches
        fields_base_directory = config.fields_base_directory

        # Step 7: Save the patches to disk in their respective temporal folders
        print("Saving patches to disk...")
        success = save_field_images_temporal(self.fields_base_directory_temporal, refined_fields)
        if success:
            print(f"Successfully saved the patches to {self.fields_base_directory_temporal}.")
        else:
            print("Failed to save the patches.")
        
        return success

    
    def get_processed_trainloader(self, batch_size, bands, vi_type='msi'):

        temporal_images = load_field_images_temporal(config.base_directory_temporal_train)
        border_removed_images_train = blacken_field_borders_temporal(temporal_images)

        if bands == 'indexbands':
            field_numbers, indices_images = indexbands_temporal_cubes(border_removed_images_train, vi_type)
        
        if bands == 'indexonly':
            field_numbers, indices_images = indexonly_temporal_cubes(border_removed_images_train, vi_type)

        if bands == 'multipleindices':
            field_numbers, indices_images = multiple_indices_temporal_cubes(border_removed_images_train)

        if bands == 'multipleindicesbands':
            field_numbers, indices_images = multiple_indices_bands_temporal_cubes(border_removed_images_train)

        if bands == 'relevantbands':
            field_numbers, indices_images = relevantbands_temporal_cubes(border_removed_images_train)

        if bands == 'allbands':
            field_numbers, indices_images = allbands_temporal_cubes(border_removed_images_train)
            
        non_temporal_images = get_non_temporal_images(indices_images)
        image_tensor_train = np.stack(non_temporal_images) 
        dataloader_train = create_data_loader(image_tensor_train, field_numbers, batch_size=batch_size, shuffle=True)

        return field_numbers, dataloader_train


    def get_processed_testloader(self, bands, batch_size):
        
        temporal_images_test = load_field_images_temporal(config.base_directory_temporal_test)
        border_removed_images_test = blacken_field_borders_temporal(temporal_images_test)

        if bands == 'indexbands':
            field_numbers, indices_images = indexbands_temporal_cubes(border_removed_images_test, vi_type)
        
        if bands == 'indexonly':
            field_numbers, indices_images = indexonly_temporal_cubes(border_removed_images_test, vi_type)

        if bands == 'multipleindices':
            field_numbers, indices_images = multiple_indices_temporal_cubes(border_removed_images_test)

        if bands == 'multipleindicesbands':
            field_numbers, indices_images = multiple_indices_bands_temporal_cubes(border_removed_images_test)

        if bands == 'relevantbands':
            field_numbers, indices_images = relevantbands_temporal_cubes(border_removed_images_test)

        if bands == 'allbands':
            field_numbers, indices_images = allbands_temporal_cubes(border_removed_images_test)
            
        non_temporal_images_test = get_non_temporal_images(indices_images_test)
        image_tensor_test = np.stack(refined_images_test) 
        dataloader_test = create_data_loader(image_tensor_test, field_numbers_test, batch_size=batch_size, shuffle=False)

        return field_numbers, dataloader_test