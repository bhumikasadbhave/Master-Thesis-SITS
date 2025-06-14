from Pipeline.temporal_preprocessing_pipeline import *

preprocessing_pipeline = PreProcessingPipelineTemporal()

preprocessing_pipeline.run_temporal_patch_save_pipeline(type='train')
preprocessing_pipeline.run_temporal_patch_save_pipeline(type='eval')

