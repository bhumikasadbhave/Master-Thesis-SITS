from Pipeline.temporal_preprocessing_pipeline import *

pipeline = PreProcessingPipelineTemporal()

pipeline.run_temporal_patch_save_pipeline(type='train')
pipeline.run_temporal_patch_save_pipeline(type='eval')