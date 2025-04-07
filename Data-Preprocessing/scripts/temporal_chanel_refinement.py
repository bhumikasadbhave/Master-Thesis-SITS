import numpy as np
from datetime import datetime
import config

# functions for creating model-ready data: with different NDVIs 

# remove masks for every function
# remove irrelevant channels

# only index
# we will pass 'ndvi' or 'msi' or 'ci' or 'evi'

# index + relevant bands 
# we will pass 'ndvi' or 'msi' or 'ci' or 'evi'

# multiple indices
# ndvi + msi + ci + evi

# multiple indices + relevant bands
# ndvi + msi + ci + evi
 
# all sentinel bands - just remove the masks and keep all sentinel bands


## MVI ##
def mvi_temporal_cubes(temporal_images):
    """ Create temporal cubes with multiple indices (MVI) """
    np.seterr(divide='ignore', invalid='ignore')
    indices = []
    field_numbers = []
    acquisition_dates = {}
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
        acquisition_dates[combined_field_no] = dates
        indices.append(temporal_indices)
    return field_numbers, acquisition_dates, indices


## B4 ##
def b4_temporal_cubes(temporal_images):
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
            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
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



def b10_temporal_cubes_with_temp_encoding(temporal_images, method='single'):
    """ Create temporal cubes with Sentinel bands and a single date embedding channel """
    cubes = []
    field_numbers = []
    acquisition_dates = {}
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
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


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



def get_single_date_embedding(date_str, ref_date='20190601.0', max_val=106):
    """ Compute a single-channel embedding for an acquisition date as a normalized scalar """

    # date_int = int(date_str.split('.')[0])  # e.g., '20190604.0' -> 20190604
    # ref_int = int(ref_date.split('.')[0])   # e.g., '20190101.0' -> 20190101
    
    # date_diff = (date_int - ref_int) / 10000  # Rough approximation of days (div by 10000 for yyyymmdd format)
    # embedding = date_diff
    
    # return embedding
    if date_str=='20190600.0' or date_str=='20190700.0' or date_str=='20190800.0' or date_str=='20190900.0':
         date_str='20190601.0'
    date = datetime.strptime(date_str, "%Y%m%d.%f")  # Parse the date string into a datetime object
    ref = datetime.strptime(ref_date, "%Y%m%d.%f")  # Parse the reference date string

    date_diff = (date - ref).days  # Get the difference in days
    print(date_diff)
    embedding = date_diff / max_val
    print(embedding)
    return embedding


def get_sin_cos_date_embedding(date_str, max_val=106):
    """ Compute a 2D cyclical encoding (sin, cos) for an acquisition date """

    if date_str=='20190600.0' or date_str=='20190700.0' or date_str=='20190800.0' or date_str=='20190900.0':
         date_str='20190601.0'

    date_int = int(date_str.split('.')[0]) 
    date_obj = datetime.strptime(str(date_int), "%Y%m%d")
    
    doy = date_obj.timetuple().tm_yday  # Integer in [1, 365/366]

    angle = 2 * np.pi * doy / max_val
    sin_doy = np.sin(angle)
    cos_doy = np.cos(angle)
    normalized_sin = (sin_doy + 1) / 2
    normalized_cos = (cos_doy + 1) / 2
    return normalized_sin, normalized_cos




## Other functions ##

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


def indexonly_temporal_cubes(temporal_images, index_name):
    """ Create temporal cubes with only the specified vegetation index """
    np.seterr(divide='ignore', invalid='ignore')
    indices = []
    field_numbers = []
    acquisition_dates = {}
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

            if index_name == 'ndvi':
                index = (nir - red) / (nir + red)
            elif index_name == 'msi':
                index = swir / nir
            elif index_name == 'ci':
                index = (nir / red) - 1
            elif index_name == 'evi':
                index = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            else:
                raise ValueError(f"Unknown index: {index_name}")

            # Min-Max Normalization
            index = np.nan_to_num(index, nan=0.0)
            index = (index - np.min(index)) / (np.max(index) - np.min(index) + 1e-6)
            temporal_indices.append(index)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        indices.append(temporal_indices)
    return field_numbers, acquisition_dates, indices


def indexbands_temporal_cubes(temporal_images, index_name):
    """ Create temporal cubes with the specified index and relevant bands """
    np.seterr(divide='ignore', invalid='ignore')
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx = 0
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

            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            green = image[..., 1]
            swir = image[..., 8]

            if index_name == 'ndvi':
                index = (nir - red) / (nir + red)
                relevant_bands = image[..., [2, 6]]
            elif index_name == 'msi':
                index = swir / nir
                relevant_bands = image[..., [6, 8]]
            elif index_name == 'ci':
                index = (nir / red) - 1
                relevant_bands = image[..., [2, 6]]
            elif index_name == 'evi':
                index = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
                relevant_bands = image[..., [0, 2, 6]]
            else:
                raise ValueError(f"Unknown index: {index_name}")

            # Min-Max Normalization
            index = np.nan_to_num(index, nan=0.0)
            index = (index - np.min(index)) / (np.max(index) - np.min(index) + 1e-6)

            #relevant_bands = image[..., [0, 1, 2, 6, 8]]  # Blue, Green, Red, NIR, SWIR
            temporal_cubes.append(np.dstack((index, relevant_bands)))
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


def multiple_indices_bands_temporal_cubes(temporal_images):
    """ Create temporal cubes with multiple indices and relevant bands (BVI) """
    np.seterr(divide='ignore', invalid='ignore')
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx = 0
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

            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            green = image[..., 1]
            swir = image[..., 8]

            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0)

            msi = swir / nir
            msi = np.nan_to_num(msi, nan=0.0)

            ci = (nir / red) - 1
            ci = np.nan_to_num(ci, nan=0.0)

            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            evi = np.nan_to_num(evi, nan=0.0)

            # Min-Max Normalization
            ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-6)
            msi = (msi - np.min(msi)) / (np.max(msi) - np.min(msi) + 1e-6)
            # ci = (ci - np.min(ci)) / (np.max(ci) - np.min(ci) + 1e-6)
            evi = (evi - np.min(evi)) / (np.max(evi) - np.min(evi) + 1e-6)

            relevant_bands = image[..., [0, 2, 6, 8]]  # Blue, Red, NIR, SWIR
            # temporal_cubes.append(np.dstack((ndvi, msi, ci, evi, relevant_bands)))
            temporal_cubes.append(np.dstack((ndvi, msi, evi, relevant_bands)))
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


def relevantbands_temporal_cubes(temporal_images):
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

            sentinel_bands = image[..., [0, 1, 2, 3, 8, 9]]  # Exclude masks (Channels 10, 11, 12)
            temporal_cubes.append(sentinel_bands)
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


def multiple_indices_allbands_temporal_cubes(temporal_images):
    """ Create temporal cubes with multiple indices and relevant bands """
    np.seterr(divide='ignore', invalid='ignore')
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx = 0
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

            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            green = image[..., 1]
            swir = image[..., 8]

            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0)

            msi = swir / nir
            msi = np.nan_to_num(msi, nan=0.0)

            ci = (nir / red) - 1
            ci = np.nan_to_num(ci, nan=0.0)

            evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            evi = np.nan_to_num(evi, nan=0.0)

            # Min-Max Normalization
            ndvi = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi) + 1e-6)
            msi = (msi - np.min(msi)) / (np.max(msi) - np.min(msi) + 1e-6)
            # ci = (ci - np.min(ci)) / (np.max(ci) - np.min(ci) + 1e-6)
            evi = (evi - np.min(evi)) / (np.max(evi) - np.min(evi) + 1e-6)

            relevant_bands = image[..., :10]  # Blue, Green, Red, NIR, SWIR
            # temporal_cubes.append(np.dstack((ndvi, msi, ci, evi, relevant_bands)))
            temporal_cubes.append(np.dstack((ndvi, msi, evi, relevant_bands)))
            dates.append(date_unique)
        field_idx+=1
        acquisition_dates[combined_field_no] = dates
        cubes.append(temporal_cubes)
    return field_numbers, acquisition_dates, cubes


############ NDVI Differences #############

def temporal_differences_with_time(temporal_images, index_name):
    """
    Create temporal cubes with normalized differences of the specified index,
    including normalization by the time interval between consecutive images.
    Args: temporal_images (list of lists): Temporal stacks of images for multiple fields. Each inner list contains the temporal images for a single field.
        index_name (str): The vegetation index to compute ('ndvi', 'msi', 'ci', 'evi').
    Returns:list of lists: Temporal cubes with normalized differences for the specified index, including time interval normalization.
            Each inner list contains the difference stack for a single field.
    """
    np.seterr(divide='ignore', invalid='ignore')
    indices = []
    
    for temporal_stack in temporal_images:
        temporal_indices = []
        acquisition_dates = []
        
        # 1. Calculate NDVI (or other index) for each image in the stack
        for image in temporal_stack:
            nir = image[..., 6]
            red = image[..., 2]
            blue = image[..., 0]
            swir = image[..., 8]
            
            # 2. Extract acquisition date 
            date_channel = image[..., -1]  
            unique_dates = np.unique(date_channel)
            unique_dates = unique_dates[unique_dates != 0]  
            date_str = str(int(unique_dates[0]))            # Convert to string (yyyymmdd)
            acquisition_dates.append(date_str)
            
            # 3. Compute the specified index
            if index_name == 'ndvi':
                index = (nir - red) / (nir + red)
            elif index_name == 'msi':
                index = swir / nir
            elif index_name == 'ci':
                index = (nir / red) - 1
            elif index_name == 'evi':
                index = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
            else:
                raise ValueError(f"Unknown index: {index_name}")
            temporal_indices.append(index)
        
        # 4. Calculate the time differences between consecutive images in days
        time_diffs = [convert_date_to_days(date) for date in acquisition_dates]
        time_diffs = np.diff(time_diffs)  
        
        # 6. Compute differences in NDVI (or other index) between consecutive images
        temporal_indices = np.array(temporal_indices)  
        differences = np.diff(temporal_indices, axis=0)  
        
        # 7. Normalize the differences by the time interval
        normalized_differences = []
        for diff, time_diff in zip(differences, time_diffs):
            if time_diff != 0: 
                diff_norm = diff / time_diff  
                diff_norm = np.nan_to_num(diff_norm, nan=0.0)
                diff_norm = (diff_norm - np.min(diff_norm)) / (np.max(diff_norm) - np.min(diff_norm))  # Min-Max normalization
            else:
                diff_norm = diff  
            normalized_differences.append(diff_norm)
        
        indices.append(normalized_differences)
    
    return indices


def b10_temporal_deltas(temporal_images):
    """Create temporal cubes with per-channel temporal deltas (differences between consecutive frames)."""
    cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx = 0

    for temporal_stack in temporal_images:
        delta_cubes = []
        dates = []

        # Get field number
        id_mask = temporal_stack[0][..., 11]
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}'
        field_numbers.append(combined_field_no)

        prev_frame = None
        prev_date = None
        for i, image in enumerate(temporal_stack):
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = date[date != 0]
            date_unique = str(date_unique[0])

            sentinel_bands = image[..., :10]  # Channels 0–9 (B10)
            if prev_frame is not None:
                delta = sentinel_bands - prev_frame
                delta_cubes.append(delta)
                dates.append(f"{prev_date}_{date_unique}")  # Optional: save which frames the diff was between
            prev_frame = sentinel_bands
            prev_date = date_unique

        field_idx += 1
        acquisition_dates[combined_field_no] = dates
        cubes.append(delta_cubes)

    return field_numbers, acquisition_dates, cubes


def temporal_vi_deltas(temporal_images, vi_type='ndvi'):
    """ Create temporal cubes with vegetation index (NDVI, EVI, or MSI) differences between consecutive frames. """
    delta_cubes = []
    field_numbers = []
    acquisition_dates = {}
    field_idx = 0

    for temporal_stack in temporal_images:
        vi_deltas = []
        dates = []

        id_mask = temporal_stack[0][..., 11]
        field_number = np.unique(id_mask)
        field_number = field_number[field_number != 0]

        if len(field_number) > 1:
            combined_field_no = '_'.join(map(str, sorted(field_number)))
        elif len(field_number) == 1:
            combined_field_no = str(field_number[0])
        else:
            combined_field_no = f'{field_idx}'
        field_numbers.append(combined_field_no)

        prev_vi = None
        prev_date = None
        for i, image in enumerate(temporal_stack):
            date_mask = image[..., -1]
            date = np.unique(date_mask)
            date_unique = str(date[date != 0][0])

            # Calculate the selected vegetation index (NDVI, EVI, MSI)
            vi = calculate_vegetation_index(image, vi_type)

            # If this is not the first frame, compute the difference
            if prev_vi is not None:
                vi_delta = vi - prev_vi
                vi_deltas.append(vi_delta)
                dates.append(f"{prev_date}_{date_unique}")  
            prev_vi = vi
            prev_date = date_unique

        field_idx += 1
        acquisition_dates[combined_field_no] = dates
        delta_cubes.append(vi_deltas)

    return field_numbers, acquisition_dates, delta_cubes


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


############ Non-temporal Images Channel Refinement #############
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
