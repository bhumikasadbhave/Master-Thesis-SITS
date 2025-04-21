import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# def align_features_sugarcontent(extracted_features, field_numbers, sugarcontent_csv_path):
#     """ Read the CSV for sugarcontent, and align the extracted features and sugarcontent values according to the field numbers.
#         Since extracted features (latents for Autoencoders) are on sub-patch level, we distribute the sugarcontent values evenly to all subpatches.
#     """

    # read csv 2 columns are needed 'FIELDUSNO' and 'sugar_content' -> create a ground_truths dictionary with fieldusno as key and sugarcontent as its value
    # extracted features is a list of all latents, which we need to use for regression
    # if latent size > 32, use pca to reduce it to 32
    # field_numbers is list of strings of format 'fieldnum1.0_fieldnum2.0_x_y' -> discard the x and y coords ie -1 and -2 after splitting
    # it is possible that there are more than 1 field numbers for a single latent as stated above, in that case copy the latent and create separate entries for all of the field numbers
    # check if size of this newly created latents and field numbers_list is equal to the length of ground_truths dictionary
    # now, create another list which will go over the field_numbers and latents, and use the ground_truths dictionary to get sugar_content for the respective field_number
    # return the 3 lists -> latents, field_numbers, sugar_contents


def align_features_sugarcontent(extracted_features, field_numbers, sugarcontent_csv_path):
    """Read the CSV for sugarcontent, and align the extracted features and sugarcontent values according to the field numbers.
    Since extracted features (latents for Autoencoders) are on sub-patch level, we distribute the sugarcontent values evenly to all subpatches.
    """

    ground_truths = {}
    df = pd.read_csv(sugarcontent_csv_path)
    df['FIELDUSNO'] = df['FIELDUSNO'].astype(str).str.strip()
    df['sugar_perc_span'] = df['sugar_perc_span'] * 100    #convert to percent
    ground_truths = dict(zip(df['FIELDUSNO'], df['sugar_perc_span']))

    #print(ground_truths)

    expanded_latents = []
    expanded_field_numbers = []

    for latent, field_str in zip(extracted_features, field_numbers):
        field_ids = field_str.split('_')[:-2]  # discard x, y
        for field_id in field_ids:
            expanded_latents.append(latent)
            expanded_field_numbers.append(field_id)

    #Dimensionality reduction if latent size > 32
    # if len(expanded_latents[0]) > 32:
    #     pca = PCA(n_components=32)
    #     expanded_latents = pca.fit_transform(expanded_latents)

    #Get same sugarcontent for all subpatches
    sugar_contents = []
    matched_latents = []
    matched_field_numbers = []

    for latent, field_id in zip(expanded_latents, expanded_field_numbers):
        field_id = str(int(float(field_id)))
        #print(field_id)
        if field_id in ground_truths:
            #print('match')
            sugar_contents.append(ground_truths[field_id])
            matched_latents.append(latent)
            matched_field_numbers.append(field_id)

    return matched_latents, matched_field_numbers, sugar_contents



