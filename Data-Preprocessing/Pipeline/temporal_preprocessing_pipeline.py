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
from model_scripts.model_helper import *
from model_scripts.dataset_creation import *
from model_scripts.train_model_ae import *
from model_scripts.model_visualiser import *
from Pipeline.temporal_preprocessing_pipeline import *


class PreProcessingPipelineTemporal:

    def __init__(self):

        #Get values from Config file
        self.sentinel_base_path = config.sentinel_base_path
        self.sentinel_base_path_eval = config.sentinel_base_path_eval
        self.load_train_dir = config.load_directory_temporal_train
        self.load_eval_dir = config.load_directory_temporal_eval
        self.save_directory_temporal_train = config.save_directory_temporal_train
        self.save_directory_temporal_eval = config.save_directory_temporal_eval
        self.field_size = config.patch_size
        self.temporal_stack_size = config.temporal_stack_size
        self.date_ranges = config.temporal_points


    def run_temporal_patch_save_pipeline(self,type='train'):
        """
        Function to run the patch-save preprocessing pipeline for temporal image stacks.
        It loads images, integrates masks, applies masking, extracts patches, and saves them.
        """

        # Step 1: Load Sentinel Images and integrate Corresponding Masks
        if type == 'train':
            images = load_sentinel_images_temporal(self.sentinel_base_path)
        elif type == 'eval':
            images = load_sentinel_images_temporal(self.sentinel_base_path_eval)
        print(f"Loaded {len(images)} temporal images and with attached masks in them.")

        # Step 2: Mask images using Sugarbeet Field ID mask
        masked_images = mask_images_temporal(images)
        print(f"Masked {len(masked_images)} images.")

        # Step 3: Extract patches from the masked temporal images 
        fields = extract_fields_temporal(masked_images, self.field_size)
        print(f"Extracted {len(fields)} patches from temporal images.")

        # Setp 4: Refine the temporal stack: 7 cloud-free images per patch with atleast 5-day gap between each successive temporal images
        refined_fields = refine_temporal_stack_interval5(fields, self.temporal_stack_size, self.date_ranges)

        # Save patches
        if type == 'train':
            fields_base_directory = self.save_directory_temporal_train
        elif type == 'eval':
            fields_base_directory = self.save_directory_temporal_eval

        print("Saving patches to disk...")
        success = save_field_images_temporal(fields_base_directory, refined_fields)
        if success:
            print(f"Successfully saved the patches to {fields_base_directory}.")
        else:
            print("Failed to save the patches.")
        return success

    
    def get_processed_temporal_cubes(self, dataset_type, bands, vi_type='msi', method='single'):
        """ 
        Generalized pipeline to load the saved field patches, remove border pixels, 
        and get final model-ready temporal cube for both train and test data.
        Returns image tensor and field numbers.
        
        Parameters:
            dataset_type (str): 'train' or 'eval' to specify dataset.
            bands (str): Type of band selection method.
            vi_type (str, optional): Type of vegetation index in case 'indexbands' OR 'indexonly' is used, default is 'msi'.
        """

        # Load the saved patches from the file system
        if dataset_type == 'train':
            temporal_images = load_field_images_temporal(self.load_train_dir)

            # Filter non-sugarbeet fields for train data
            temporal_images = filter_non_sugarbeet_fields(temporal_images, config.sugarbeet_content_csv_path)

        elif dataset_type == 'eval':
            temporal_images = load_field_images_temporal(self.load_eval_dir)
        else:
            raise ValueError("dataset_type must be either 'train' or 'test'")
        

        # Setp 4: Remove the border pixels of the sugarbeet fields
        border_removed_images = blacken_field_borders_temporal(temporal_images)

        # Normalize images
        normalized_images = normalize_images(border_removed_images)

        # Step 5: Select relevant Vegetation Indices and Sentinel-2 Bands
        band_selection_methods = {
            'rgb': rgb_temporal_cubes,
            'mvi': mvi_temporal_cubes,
            'b4': b4_temporal_cubes,
            'b10': b10_temporal_cubes,
            'b10_channel': b10_temporal_cubes_with_temp_encoding,       # temporal encodings as 2 extra channels
            'b10_add': b10_temporal_cubes_with_temp_encoding_returned,   # temporal encodings returned for addition in autoencoder
            'b4_add': b4_temporal_cubes_with_temp_encoding_returned,   # temporal encodings returned for addition in autoencoder
            'mvi_add': mvi_temporal_cubes_with_temp_encoding_returned   # temporal encodings returned for addition in autoencoder
        }

        if bands not in band_selection_methods:
            raise ValueError(f"Invalid bands option: {bands}")

        if bands in ['indexbands', 'indexonly', 'vid']:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images, vi_type)
        elif bands in ['b10_channel','b4_channel']:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images, method)
        elif bands in ['b10_add', 'b4_add', 'mvi_add']:
            field_numbers, acquisition_dates, date_emb, indices_images = band_selection_methods[bands](normalized_images, method)
        else:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images)

        # Step 5: Return Temporal Cubes for training, and list of images for visualisation
        images_visualisation = normalized_images
        image_tensor = np.stack(indices_images)
        if bands != 'vid':
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(0, 1, 4, 2, 3) # N, T, C, H, W
        
        if bands in ['b10_add', 'b4_add', 'mvi_add']:
            return field_numbers, acquisition_dates, date_emb, image_tensor, images_visualisation
        return field_numbers, acquisition_dates, image_tensor, images_visualisation


    def get_processed_non_temporal_data(self, dataset_type, bands, vi_type='msi', method='single'):
        """ 
        Pipeline to load the saved field patches, remove border pixels, 
        and extract the last image from the temporal stack as non-temporal data.
        Returns image tensor and field numbers.
        
        Parameters:
            dataset_type (str): 'train' or 'eval' to specify dataset.
            bands (str): Type of band selection method.
            vi_type (str, optional): Type of vegetation index in case 'indexbands' OR 'indexonly' is used, default is 'msi'.
        """

        # Load the saved patches from the file system
        if dataset_type == 'train':
            temporal_images = load_field_images_temporal(self.load_train_dir)

            # Filter non-sugarbeet fields for train data
            temporal_images = filter_non_sugarbeet_fields(temporal_images, config.sugarbeet_content_csv_path)

        elif dataset_type == 'eval':
            temporal_images = load_field_images_temporal(self.load_eval_dir)
        else:
            raise ValueError("dataset_type must be either 'train' or 'eval'")

        # Step 4: Remove border pixels
        border_removed_images = blacken_field_borders_temporal(temporal_images)

        # Normalize
        normalized_images = normalize_images(border_removed_images)

        # Step 5: Select relevant Vegetation Indices and Sentinel-2 Bands
        band_selection_methods = {
            'rgb': rgb_temporal_cubes,
            'mvi': mvi_temporal_cubes,
            'b4': b4_temporal_cubes,
            'b10': b10_temporal_cubes,
            'b10_channel': b10_temporal_cubes_with_temp_encoding,       # temporal encodings as 2 extra channels
            'b10_add': b10_temporal_cubes_with_temp_encoding_returned,   # temporal encodings returned for addition in autoencoder
            'b4_add': b4_temporal_cubes_with_temp_encoding_returned,   # temporal encodings returned for addition in autoencoder
            'mvi_add': mvi_temporal_cubes_with_temp_encoding_returned   # temporal encodings returned for addition in autoencoder
        }

        if bands not in band_selection_methods:
            raise ValueError(f"Invalid bands option: {bands}")

        if bands in ['indexbands', 'indexonly', 'vid']:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images, vi_type)
        elif bands in ['b10_channel','b4_channel']:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images, method)
        elif bands in ['b10_add', 'b4_add', 'mvi_add']:
            field_numbers, acquisition_dates, date_emb, indices_images = band_selection_methods[bands](normalized_images, method)
        else:
            field_numbers, acquisition_dates, indices_images = band_selection_methods[bands](normalized_images)

        # Step 5: Extract the last image from each temporal stack
        non_temporal_images = [stack[-1] for stack in indices_images]  #Selecting the last image (September image)

        # Convert to tensor format for modelling, and return list of images for Visualisation
        images_visualisation = non_temporal_images
        image_tensor = np.stack(non_temporal_images)
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(0, 3, 1, 2)  # No temporal dimension

        if bands in ['b10_add', 'b4_add', 'mvi_add']:
            return field_numbers, acquisition_dates, date_emb, image_tensor, images_visualisation
        return field_numbers, acquisition_dates, image_tensor, images_visualisation

    # For masked autoencoder
    def get_processed_temporal_cube3(self, dataset_type, bands, vi_type='msi'):
            """ 
            Pipeline to load the saved field patches, remove border pixels, 
            and extract the last image from the temporal stack as non-temporal data.
            Returns image tensor and field numbers.
            
            Parameters:
                dataset_type (str): 'train' or 'eval' to specify dataset.
                bands (str): Type of band selection method.
                vi_type (str, optional): Type of vegetation index in case 'indexbands' OR 'indexonly' is used, default is 'msi'.
            """

            # Step 1: Load the saved patches from the file system
            if dataset_type == 'train':
                temporal_images = load_field_images_temporal(self.load_train_dir)

                # Filter non-sugarbeet fields for train data
                temporal_images = filter_non_sugarbeet_fields(temporal_images, config.sugarbeet_content_csv_path)
                
            elif dataset_type == 'eval':
                temporal_images = load_field_images_temporal(self.load_eval_dir)
            else:
                raise ValueError("dataset_type must be either 'train' or 'eval'")

            # Step 2: Remove border pixels
            border_removed_images = blacken_field_borders_temporal(temporal_images)

            # Step 3: Normalize
            normalized_images = normalize_images(border_removed_images)

            # Step 4: Select relevant Vegetation Indices and Sentinel-2 Bands
            band_selection_methods = {
                'rgb': rgb_temporal_cubes,
                'mvi': mvi_temporal_cubes,
                'b4': b4_temporal_cubes,
                'b10': b10_temporal_cubes
            }

            if bands not in band_selection_methods:
                raise ValueError(f"Invalid bands option: {bands}")

            if bands in ['indexbands', 'indexonly']:
                field_numbers, acquisition_dates, processed_images = band_selection_methods[bands](normalized_images, vi_type)
            else:
                field_numbers, acquisition_dates, processed_images = band_selection_methods[bands](normalized_images)

            selected_images = []
            selected_acquisition_dates = []
            
            for i, stack in enumerate(processed_images):
                selected_images.append([stack[2], stack[4], stack[6]])
                # field_num = int(float(field_numbers[i]))
                # selected_acquisition_dates[field_num] = [acquisition_dates[field_numbers[i]][j] for j in [2, 4, 6]]
                # print(selected_acquisition_dates)

            # Step 5: Return Temporal Cubes for training, and list of images for visualisation
            images_visualisation = selected_images
            image_tensor = np.stack(selected_images)
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(0, 1, 4, 2, 3)
            
            return field_numbers, acquisition_dates, image_tensor, images_visualisation



