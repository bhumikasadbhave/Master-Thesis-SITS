from Pipeline.preprocess_script import *
from Pipeline.temporal_preprocessing_pipeline import *

# Extract sugarbeet fields as patches and save them on the file system
pipeline = PreProcessingPipelineTemporal()
pipeline.run_temporal_patch_save_pipeline(type='train')
pipeline.run_temporal_patch_save_pipeline(type='eval')

# Get pre-processed B10 data tensors
dataloader_train, dataloader_test, dataloader_eval = get_model_ready_data(model_type='autoencoders_addition', tensor_type='b10_add', encoding_method='sin-cos', visualisation_images=False)

# Model the data using 3D_AE_B10 with temporal encodings
