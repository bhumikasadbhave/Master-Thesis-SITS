import numpy as np
from datetime import datetime
import config

# functions for creating model-ready data

## --- Data cubes without temporal encodings --- ##

## MVI ##
def mvi_temporal_cubes(temporal_images):
    """ Create temporal cubes with multiple indices (MVI) """
    np.seterr(divide='ignore', invalid='ignore')
    indices = []
    field_numbers = []
    acquisition_dates = []
    field_idx = 0
    for temporal_stack in temporal_images:
        temporal_indices = []
        dates = []

        #Get field number
        id_mask = temporal_stack[0][..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:

            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            swir = image[..., 8]

            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0)

            msi = swir / nir
            msi = np.nan_to_num(msi, nan=0.0)

            # ci = (nir / red) - 1
            # ci = np.nan_to_num(ci, nan=0.0)

            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            evi = np.nan_to_num(evi, nan=0.0)

            # Min-Max Normalization
            ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-6)
            msi = (msi - np.min(msi)) / (np.max(msi) - np.min(msi) + 1e-6)
            # ci = (ci - np.min(ci)) / (np.max(ci) - np.min(ci) + 1e-6)
            evi = (evi - np.min(evi)) / (np.max(evi) - np.min(evi) + 1e-6)

            # print(np.unique(ndvi))

            # temporal_indices.append(np.dstack((ndvi, msi, ci, evi)))
            temporal_indices.append(np.dstack((ndvi, msi, evi)))
            dates.append(date_unique)

        field_idx+=1
        acquisition_dates.append(dates)
        indices.append(temporal_indices)
    return field_numbers, acquisition_dates, indices


## B4 ##
def b4_temporal_cubes(temporal_images):
    """ Create temporal cubes with all Sentinel bands excluding masks """
    cubes = []
    field_numbers = []
    acquisition_dates = []
    field_idx=0
    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []

        #Get field number
        id_mask = temporal_stack[0][..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:

            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., [0, 2, 6, 8]]  # Only channels used for calculating NDVI, EVI, MSI
            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates.append(dates)
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


## B10 ##
def b10_temporal_cubes(temporal_images):
    """ Create temporal cubes with all Sentinel bands excluding masks (B10) """
    cubes = []
    field_numbers = []
    acquisition_dates = []
    field_idx=0
    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []

        #Get field number
        id_mask = temporal_stack[0][..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:

            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., :10]  # Exclude masks (Channels 10, 11, 12)
            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates.append(dates)
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


## --- Temporal Encodings as channels --- ##

## B10 ##
def b10_temporal_cubes_with_temp_encoding(temporal_images, method='single'):
    """ Create temporal cubes with Sentinel bands and a single date embedding channel """
    cubes = []
    field_numbers = []
    acquisition_dates = []
    field_idx = 0
    
    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []

        # Get field number
        id_mask = temporal_stack[0][..., 11]  # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])  

            sentinel_bands = image[..., :10]  #Shape: (H, W, 10)

            # Compute single date embedding -> Add as addiional channel...
            if method == 'single':
                date_embedding = get_single_date_embedding(date_unique, ref_date='20190601.0')  #Scalar
                h, w = config.patch_size
                embedding_channel = np.full((h, w, 1), date_embedding)  #Shape: (H, W, 1)
                augmented_bands = np.dstack((sentinel_bands, embedding_channel))  # Shape: (H, W, 11)

            elif method == 'sin-cos':
                date_embedding_sin, date_embedding_cos = get_sin_cos_date_embedding(date_unique)  #Scalar
                h, w = config.patch_size
                embedding_channel1 = np.full((h, w, 1), date_embedding_sin)  #Shape: (H, W, 1)
                embedding_channel2 = np.full((h, w, 1), date_embedding_cos)  #Shape: (H, W, 1)
                augmented_bands = np.dstack((sentinel_bands, embedding_channel1, embedding_channel2))  # Shape: (H, W, 12)

            temporal_cubes.append(augmented_bands)
            dates.append(date_unique)
        field_idx += 1
        acquisition_dates.append(dates)
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


## B4 ##
def b4_temporal_cubes_with_temp_encoding(temporal_images, method='single'):
    """ Create temporal cubes with all Sentinel bands excluding masks """
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx=0
    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []

        #Get field number
        id_mask = temporal_stack[0][..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:

            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., [0, 2, 6, 8]]  # Only channels used for calculating NDVI, EVI, MSI

            # Compute single date embedding -> Add as addiional channel...
            if method == 'single':
                date_embedding = get_single_date_embedding(date_unique, ref_date=config.ref_date, max_val=config.max_date_diff)  #Scalar
                h, w = config.patch_size
                embedding_channel = np.full((h, w, 1), date_embedding)  #Shape: (H, W, 1)
                augmented_bands = np.dstack((sentinel_bands, embedding_channel))  # Shape: (H, W, 11)

            elif method == 'sin-cos':
                date_embedding_sin, date_embedding_cos = get_sin_cos_date_embedding(date_unique)  #Scalar
                h, w = config.patch_size
                embedding_channel1 = np.full((h, w, 1), date_embedding_sin)  #Shape: (H, W, 1)
                embedding_channel2 = np.full((h, w, 1), date_embedding_cos)  #Shape: (H, W, 1)
                augmented_bands = np.dstack((sentinel_bands, embedding_channel1, embedding_channel2))  # Shape: (H, W, 12)

            temporal_cubes.append(augmented_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


## --- Temporal Encodings as addition in forward - returns date embeddings --- ##

## B10 ##
def b10_temporal_cubes_with_temp_encoding_returned(temporal_images, method='sin-cos'):
    """Create temporal cubes with Sentinel bands by adding date embedding values directly to bands"""
    cubes = []
    field_numbers = []
    acquisition_dates = []
    all_date_embeddings = []
    field_idx = 0

    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []
        date_embeddings = []

        # Get field number
        id_mask = temporal_stack[0][..., 11]  # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}'
        field_numbers.append(combined_field_no)

        for image in temporal_stack:
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., :10]  # Shape: (H, W, 10)
            # print(date_unique)

            # --- Temporal encoding to be ADDED to existing pixel values ---
            if method == 'single':
                date_embedding = get_single_date_embedding(date_unique, ref_date='20190601.0')  # scalar
                # encoded = sentinel_bands + date_embedding
                date_embeddings.append(date_embedding)

            elif method == 'sin-cos':
                sin_emb, cos_emb = get_sin_cos_date_embedding(date_unique)                      # both scalars
                # encoded = sentinel_bands + sin_emb + cos_emb
                date_embeddings.append([sin_emb,cos_emb])

            else:
                raise ValueError(f"Unknown method '{method}' for temporal encoding.")

            # --- Re-normalize back to [0, 1] ---
            # encoded = np.clip(encoded, 0, None)  # avoid negatives before normalization
            # min_val = np.min(encoded)
            # max_val = np.max(encoded)
            # normalized_encoded = (encoded - min_val) / (max_val - min_val + 1e-8)

            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)

        field_idx += 1
        acquisition_dates.append(dates)
        all_date_embeddings.append(date_embeddings)
        cubes.append(temporal_cubes)

    return field_numbers, acquisition_dates, all_date_embeddings, cubes


## B4 ##
def b4_temporal_cubes_with_temp_encoding_returned(temporal_images, method='sin-cos'):
    """Create temporal cubes with Sentinel bands by adding date embedding values directly to bands"""
    cubes = []
    field_numbers = []
    acquisition_dates = []
    all_date_embeddings = []
    field_idx = 0

    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []
        date_embeddings = []

        # Get field number
        id_mask = temporal_stack[0][..., 11]  # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}'
        field_numbers.append(combined_field_no)

        for image in temporal_stack:
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., [0, 2, 6, 8]]  # Only channels used for calculating NDVI, EVI, MSI

            # --- Temporal encoding to be ADDED to existing pixel values ---
            if method == 'single':
                date_embedding = get_single_date_embedding(date_unique, ref_date='20190601.0')  # scalar
                # encoded = sentinel_bands + date_embedding
                date_embeddings.append(date_embedding)

            elif method == 'sin-cos':
                sin_emb, cos_emb = get_sin_cos_date_embedding(date_unique)                      # both scalars
                # encoded = sentinel_bands + sin_emb + cos_emb
                date_embeddings.append([sin_emb,cos_emb])

            else:
                raise ValueError(f"Unknown method '{method}' for temporal encoding.")

            # --- Re-normalize back to [0, 1] ---
            # encoded = np.clip(encoded, 0, None)  # avoid negatives before normalization
            # min_val = np.min(encoded)
            # max_val = np.max(encoded)
            # normalized_encoded = (encoded - min_val) / (max_val - min_val + 1e-8)

            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)

        field_idx += 1
        acquisition_dates.append(dates)
        all_date_embeddings.append(date_embeddings)
        cubes.append(temporal_cubes)

    return field_numbers, acquisition_dates, all_date_embeddings, cubes


## MVI ##
def mvi_temporal_cubes_with_temp_encoding_returned(temporal_images, method='sin-cos'):
    """Create temporal cubes with Sentinel bands by adding date embedding values directly to bands"""
    cubes = []
    field_numbers = []
    acquisition_dates = []
    all_date_embeddings = []
    field_idx = 0

    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []
        date_embeddings = []

        # Get field number
        id_mask = temporal_stack[0][..., 11]  # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}'
        field_numbers.append(combined_field_no)

        for image in temporal_stack:
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            swir = image[..., 8]

            ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
            ndvi = np.nan_to_num(ndvi)  

            msi = np.where(nir != 0, swir / nir, 0)
            msi = np.nan_to_num(msi)

            # ndvi = (nir - red) / (nir + red)
            # ndvi = np.nan_to_num(ndvi, nan=0.0)

            # msi = swir / nir
            # msi = np.nan_to_num(msi, nan=0.0)

            evi = np.where((nir + red) != 0, 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1), 0)
            evi = np.nan_to_num(evi, nan=0.0)

            # Min-Max Normalization
            ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-6)
            msi = (msi - np.min(msi)) / (np.max(msi) - np.min(msi) + 1e-6)
            evi = (evi - np.min(evi)) / (np.max(evi) - np.min(evi) + 1e-6)

            vi_bands = (np.dstack((ndvi, msi, evi)))   # VIs, Shape: (H, W, 10)

            # --- Temporal encoding to be ADDED to existing pixel values ---
            if method == 'single':
                date_embedding = get_single_date_embedding(date_unique, ref_date=config.reference_date_temp_encoding)  # scalar
                date_embeddings.append(date_embedding)

            elif method == 'sin-cos':
                sin_emb, cos_emb = get_sin_cos_date_embedding(date_unique)                      # both scalars
                date_embeddings.append([sin_emb,cos_emb])

            else:
                raise ValueError(f"Unknown method '{method}' for temporal encoding.")

            # --- Re-normalize back to [0, 1] ---
            # encoded = np.clip(encoded, 0, None)  # avoid negatives before normalization
            # min_val = np.min(encoded)
            # max_val = np.max(encoded)
            # normalized_encoded = (encoded - min_val) / (max_val - min_val + 1e-8)

            temporal_cubes.append(vi_bands)
            dates.append(date_unique)

        field_idx += 1
        acquisition_dates.append(dates)
        all_date_embeddings.append(date_embeddings)
        cubes.append(temporal_cubes)

    return field_numbers, acquisition_dates, all_date_embeddings, cubes



## --- Utility Functions --- ##

def get_single_date_embedding(date_str, ref_date='20190601.0', max_val=106):
    """ Compute a single-channel embedding for an acquisition date as a normalized scalar """

    # date_int = int(date_str.split('.')[0])  # e.g., '20190604.0' -> 20190604
    # ref_int = int(ref_date.split('.')[0])   # e.g., '20190101.0' -> 20190101
    
    # date_diff = (date_int - ref_int) / 10000  # Rough approximation of days (div by 10000 for yyyymmdd format)
    # embedding = date_diff
    
    # return embedding
    if date_str=='20190600.0' or date_str=='20190700.0' or date_str=='20190800.0' or date_str=='20190900.0':
         date_str='20190601.0'
    
    if date_str=='20240600.0' or date_str=='20240700.0' or date_str=='20240800.0' or date_str=='20240900.0':
         date_str='20240601.0'

    date = datetime.strptime(date_str, "%Y%m%d.%f")  # Parse the date string into a datetime object
    ref = datetime.strptime(ref_date, "%Y%m%d.%f")  # Parse the reference date string

    date_diff = (date - ref).days  # Get the difference in days
    # print(date_diff)
    embedding = date_diff / max_val
    # print(embedding)
    return embedding


def get_sin_cos_date_embedding(date_str, max_val=106):
    """ Compute a 2D cyclical encoding (sin, cos) for an acquisition date """

    if date_str=='20190600.0' or date_str=='20190700.0' or date_str=='20190800.0' or date_str=='20190900.0':
         date_str='20190601.0'

    if date_str=='20240600.0' or date_str=='20240700.0' or date_str=='20240800.0' or date_str=='20240900.0':
         date_str='20240601.0'

    date_int = int(date_str.split('.')[0]) 
    # print(date_str)
    date_obj = datetime.strptime(str(date_int), "%Y%m%d")
    
    doy = date_obj.timetuple().tm_yday  # Integer in [1, 365/366]

    angle = 2 * np.pi * doy / max_val
    sin_doy = np.sin(angle)
    cos_doy = np.cos(angle)
    return sin_doy, cos_doy


## --- Non-temporal Images Channel Refinement --- ##
# Refine september images -- Keep IMP Bands only 0,1,2,3,8,9
def refine_chanel_non_temporal(images):

    refined_images = []
    field_numbers = []
    field_idx = 0

    for image in images:

        id_mask = image[..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 

        field_numbers.append(combined_field_no)
        field_idx += 1

        relevant_bands = image[..., [0, 1, 2, 3, 8, 9]]
        refined_images.append(relevant_bands)

    return field_numbers, refined_images


## --- Other functions --- ##
# Used for pre-trained models - RGB channels only
def rgb_temporal_cubes(temporal_images):
    """ Create temporal cubes with all Sentinel bands excluding masks """
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx=0
    for temporal_stack in temporal_images:
        temporal_cubes = []
        dates = []

        #Get field number
        id_mask = temporal_stack[0][..., 11]                   # field_id
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]  

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}' 
        field_numbers.append(combined_field_no)

        for image in temporal_stack:

            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., [0, 1, 2]]  # Exclude masks (Channels 10, 11, 12)
            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes

# VI calculation
def calculate_vegetation_index(image, vi_type='ndvi'):
    """ Calculate the vegetation index for a single image based on the selected index type. """

    nir = image[..., 6]
    red = image[..., 2]
    blue = image[..., 0]
    swir = image[..., 8]
    
    if vi_type == 'ndvi':
        # NDVI formula: (NIR - Red) / (NIR + Red)
        return (nir - red) / (nir + red + 1e-8)
    
    elif vi_type == 'evi':
        # EVI formula: 2.5 * (NIR - Red) / (NIR + 6*Blue - 7.5*Red + 10000)
        G = 2.5
        C1 = 6
        C2 = 7.5
        L = 10000
        return G * (nir - red) / (nir + C1 * blue - C2 * red + L + 1e-8)
    
    elif vi_type == 'msi':
        # MSI formula: B11 / B8
        return swir / (nir + 1e-8) 
    
    else:
        raise ValueError("Invalid vegetation index type. Choose from 'ndvi', 'evi', or 'msi'.")

# Helper Function
def convert_date_to_days(date_str):
    """Convert yyyymmdd string to number of days since the epoch (1970-01-01)."""
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    return (date_obj - datetime(1970, 1, 1)).days

