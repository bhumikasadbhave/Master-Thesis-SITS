from Pipeline.temporal_preprocessing_pipeline import *
from model_scripts.subpatch_extraction import *
from evaluation_scripts.evaluation_helper import *

preprocessing_pipeline = PreProcessingPipelineTemporal()


# Getting pre-processed sub-patches
def get_model_ready_data(model_type='baseline', tensor_type='b10', encoding_method='sin-cos', visualisation_images=False):

    # works with b10
    if model_type == 'baseline':
        field_numbers_train, acquisition_dates_train, patch_tensor_train, visualisation_train = preprocessing_pipeline.get_processed_temporal_cubes('train', tensor_type)
        field_numbers_eval, acquisition_dates_eval, patch_tensor_eval, visualisation_eval = preprocessing_pipeline.get_processed_temporal_cubes('eval', tensor_type)
        train_subpatches, train_subpatch_coords = non_overlapping_sliding_window(patch_tensor_train, field_numbers_train, patch_size=config.subpatch_size)
        eval_subpatches, eval_subpatch_coords = non_overlapping_sliding_window(patch_tensor_eval, field_numbers_eval, patch_size=config.subpatch_size)
        train_coord_fn = get_string_fielddata(train_subpatch_coords)
        eval_coord_fn = get_string_fielddata(eval_subpatch_coords)
        return train_subpatches, eval_subpatches, train_coord_fn, eval_coord_fn
    
    # works with b10
    elif model_type == 'autoencoders':
        field_numbers_train, acquisition_dates_train, patch_tensor_train, images_visualisation_train = preprocessing_pipeline.get_processed_temporal_cubes('train', tensor_type, method=encoding_method)
        field_numbers_eval, acquisition_dates_eval, patch_tensor_eval, images_visualisation_eval = preprocessing_pipeline.get_processed_temporal_cubes('eval', tensor_type, method=encoding_method)
        train_subpatches, train_subpatch_coords = non_overlapping_sliding_window(patch_tensor_train, field_numbers_train, patch_size=config.subpatch_size)
        
        if visualisation_images:
            old_images_train = {fn: img for fn, img in zip(field_numbers_train, images_visualisation_train)}
            old_images_eval = {fn: img for fn, img in zip(field_numbers_eval, images_visualisation_eval)}
            acq_dict_train = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_train)}
            acq_dict_eval = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_eval)}
            return old_images_train, old_images_eval, acq_dict_train, acq_dict_eval
        
        eval_subpatches, eval_subpatch_coords = non_overlapping_sliding_window(patch_tensor_eval, field_numbers_eval, patch_size=config.subpatch_size)
        train_coord_fn = get_string_fielddata(train_subpatch_coords)
        eval_coord_fn = get_string_fielddata(eval_subpatch_coords)
        train_subpatches_dl, test_subpatches, train_field_numbers_dl, test_field_numbers_dl = train_test_split(
            train_subpatches, train_coord_fn, test_size=1-config.ae_train_test_ratio, random_state=42)
        dataloader_train = create_data_loader(train_subpatches_dl, train_field_numbers_dl, batch_size=config.ae_batch_size, shuffle=True)
        dataloader_test = create_data_loader(test_subpatches, test_field_numbers_dl, batch_size=config.ae_batch_size, shuffle=False)
        dataloader_eval = create_data_loader(eval_subpatches, eval_coord_fn, batch_size=config.ae_batch_size, shuffle=False)
        return dataloader_train, dataloader_test, dataloader_eval

    # works with b10_channel
    elif model_type == 'autoencoders_channel':
        field_numbers_train, acquisition_dates_train, patch_tensor_train, images_visualisation_train = preprocessing_pipeline.get_processed_temporal_cubes('train', tensor_type, method=encoding_method)
        field_numbers_eval, acquisition_dates_eval, patch_tensor_eval, images_visualisation_eval = preprocessing_pipeline.get_processed_temporal_cubes('eval', tensor_type, method=encoding_method)
        train_subpatches, train_subpatch_coords = non_overlapping_sliding_window(patch_tensor_train, field_numbers_train, patch_size=config.subpatch_size, num_encoding_channels=config.num_encoding_channels)
        eval_subpatches, eval_subpatch_coords = non_overlapping_sliding_window(patch_tensor_eval, field_numbers_eval, patch_size=config.subpatch_size, num_encoding_channels=config.num_encoding_channels)
        
        if visualisation_images:
            old_images_train = {fn: img for fn, img in zip(field_numbers_train, images_visualisation_train)}
            old_images_eval = {fn: img for fn, img in zip(field_numbers_eval, images_visualisation_eval)}
            acq_dict_train = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_train)}
            acq_dict_eval = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_eval)}
            return old_images_train, old_images_eval, acq_dict_train, acq_dict_eval
        
        train_coord_dataloader = get_string_fielddata(train_subpatch_coords)
        eval_coord_dataloader = get_string_fielddata(eval_subpatch_coords)
        train_subpatches_dl, test_subpatches, train_field_numbers_dl, test_field_numbers_dl = train_test_split(
            train_subpatches, train_coord_dataloader, test_size=1-config.ae_train_test_ratio, random_state=42)
        dataloader_train = create_data_loader(train_subpatches_dl, train_field_numbers_dl, batch_size=config.ae_batch_size, shuffle=True)
        dataloader_test = create_data_loader(test_subpatches, test_field_numbers_dl, batch_size=config.ae_batch_size, shuffle=False)
        dataloader_eval = create_data_loader(eval_subpatches, eval_coord_dataloader, batch_size=config.ae_batch_size, shuffle=False)
        return dataloader_train, dataloader_test, dataloader_eval
    
    # works with b10_add, mvi_add, b4_add
    elif model_type == 'autoencoders_addition':
        field_numbers_train, acquisition_dates_train, date_emb_train, patch_tensor_train, images_visualisation_train = preprocessing_pipeline.get_processed_temporal_cubes('train', tensor_type, method=encoding_method)
        field_numbers_eval, acquisition_dates_eval, date_emb_eval, patch_tensor_eval, images_visualisation_eval = preprocessing_pipeline.get_processed_temporal_cubes('eval', tensor_type, method=encoding_method)
        train_subpatches, train_subpatch_coords, train_subpatch_date_emb = non_overlapping_sliding_window_with_date_emb(patch_tensor_train, field_numbers_train, date_emb_train, patch_size=config.subpatch_size)
        eval_subpatches, eval_subpatch_coords, eval_subpatch_date_emb = non_overlapping_sliding_window_with_date_emb(patch_tensor_eval, field_numbers_eval, date_emb_eval, patch_size=config.subpatch_size)

        if visualisation_images:
            old_images_train = {fn: img for fn, img in zip(field_numbers_train, images_visualisation_train)}
            old_images_eval = {fn: img for fn, img in zip(field_numbers_eval, images_visualisation_eval)}
            acq_dict_train = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_train)}
            acq_dict_eval = {fn: date for fn, date in zip(field_numbers_train, acquisition_dates_eval)}
            return old_images_train, old_images_eval, acq_dict_train, acq_dict_eval

        train_coord_dataloader = get_string_fielddata(train_subpatch_coords)
        eval_coord_dataloader = get_string_fielddata(eval_subpatch_coords)
        train_subpatches_dl, test_subpatches, train_field_numbers_dl, test_field_numbers_dl, train_date_embeddings, test_date_embeddings = train_test_split(
            train_subpatches, train_coord_dataloader, train_subpatch_date_emb, test_size=1-config.ae_train_test_ratio, random_state=42)
        dataloader_train = create_data_loader_mae(train_subpatches_dl, train_field_numbers_dl, train_date_embeddings, mae=False, batch_size=config.ae_batch_size, shuffle=True)
        dataloader_test = create_data_loader_mae(test_subpatches, test_field_numbers_dl, test_date_embeddings, mae=False, batch_size=config.ae_batch_size, shuffle=False)
        dataloader_eval = create_data_loader_mae(eval_subpatches, eval_coord_dataloader, eval_subpatch_date_emb, mae=False, batch_size=config.ae_batch_size, shuffle=False)
        return dataloader_train, dataloader_test, dataloader_eval

    else:
        return "Please put the correct method name"
    