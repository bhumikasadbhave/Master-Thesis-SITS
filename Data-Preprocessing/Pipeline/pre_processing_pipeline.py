from scripts.data_loader import *
from scripts.data_preprocessor import *
from scripts.data_visualiser import *
from scripts.preprocess_helper import *
import config as config

class PreProcessingPipeline:

    def __init__(self):
        self.sentinel_image_path = config.sentinel_image_directory
        self.sentinel_mask_path = config.sentinel_mask_directory
        self.sentinel_id_path = config.sentinel_id_directory
        self.fields_base_directory = config.fields_base_directory
        self.field_size = config.field_size
        self.field_size = config.field_size
        self.train_test_ratio = config.train_test_ratio

    def run_full_preprocessing_pipeline(self):
        
        #Load Raw Sentinel Images and Sugar-beet Masks
        sentinel_images, sugarbeet_masks, id_masks = load_sentinel_images(self.sentinel_image_path, self.sentinel_mask_path, self.sentinel_id_path)
        print("Sugar-beet fields and masks are loaded successfully! Great start! Shape of the image is: ", sentinel_images[0].shape)

        #Add the sugarbeet mask as a new channel to the images
        integrated_images = integrate_sugarbeet_mask(sentinel_images, sugarbeet_masks)
        print("\nMasks are combined with the images successfully! The shape of image is now: ", integrated_images[0].shape)

        #Mask the pixels that are covered by Cloud or that do not belong to sugar-beet fields
        print("\nMasking pixels that are cloud-covered or that don't belong to sugarbeet fields ... ")
        masked_images = mask_images(integrated_images)
        print("Successfully gotten rid of unwanted pixels!")

        #Extract Sugar-beet fields from the Images
        print("\nExtracting sugar-beet fields from images ...")
        extracted_patches = extract_fields(masked_images, (64, 64))
        print("Fields extracted!")

        #Save extracted fields and create your dataset
        success = save_field_images(self.fields_base_directory, extracted_patches, self.train_test_ratio)
        if success:
            print("\nTrain and Test Dataset created for modelling! Pre-processing pipeline finished!")
        else:
            raise ValueError("Pre-processing failed! Check logs!")
        


