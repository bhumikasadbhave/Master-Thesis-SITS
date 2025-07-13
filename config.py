### --- Paths for field extraction and saving from 1000x1000 images --- ###
## 2019 data ##
sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/images-train/'
sentinel_base_path_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/images-eval/'
save_directory_temporal_train = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/patches-train'
save_directory_temporal_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/patches-eval'

#### --- Load Paths --- ####
## 2019 data ##
load_directory_temporal_train = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/patches-train'
load_directory_temporal_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/patches-eval'
deliverable_images_save_path = '/Users/bhumikasadbhave007/Desktop/THESIS/Thesis_OP'
labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/images-eval/labels.csv'
sugarbeet_content_csv_path = "/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/2019_sugar_content.csv"  #### Filtering fields that are not sugarbeet 

#### --- Temporal Parameters --- ####
# 2019
temporal_points = [
    ("june", "2019-06-01", "2019-06-30"),
    ("july", "2019-07-01", "2019-07-31"),
    ("august", "2019-08-01", "2019-08-30"),
    ("september", "2019-09-01", "2019-09-15")
]
reference_date_temp_encoding='20190601.0'
temporal_stack_size = 7

# 2024
temporal_points_2024 = [
    ("june", "2024-06-01", "2024-06-30"),
    ("july", "2024-07-01", "2024-07-31"),
    ("august", "2024-08-01", "2024-08-30"),
    ("september", "2024-09-01", "2024-09-15")
]
temporal_stack_size_2024 = 4

#### --- Other Parameters --- ####
patch_size = (64, 64)
subpatch_size = 4
batch_size=64
subpatch_to_patch_threshold = 0.5
num_encoding_channels = 2
max_date_diff = 115
ae_batch_size = 64
ae_train_test_ratio = 0.8
pca_components = 3

#### --- Saved Model Paths --- ####
#### Baselines ####
kmeans_b10_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/baselines/kmeans_b10.pkl'
kmedoids_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/baselines/kmedoids_b10.pkl'
agg_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/baselines/agg_b10.pkl'
kmeans_hist_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/baselines/kmeans_hist.pkl'
kmeans_pca_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/baselines/kmeans_pca.pkl'

#### Autoencoders main ####
kmeans_ae_3D_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/kmeans_ae_3D.pkl'
kmeans_ae_2D_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/kmeans_ae_2D.pkl'
ae_3d_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/ae_3D.pkl'
ae_2D_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/ae_2D.pkl'
mae_save_dir = '/home/k64835/'

#### Autoencoder with Temporal Encodings ####
kmeans_ae_3D_TE_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/kmeans_ae_3D_TS.pkl'
ae_3d_TE_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/ae_3D_TE.pkl'
kmeans_ae_3D_TEadd_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/kmeans_ae_3D_TSadd.pkl'
ae_3d_TEadd_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/ae_3D_TEadd.pkl'
ae_2D_TE_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/ae_2D_TE.pkl'
kmeans_ae_2D_TE_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/autoencoders/kmeans_ae_2D_TE.pkl'

#### Result JSONs for all AE-based models and experiments ####
results_json_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/GITHUB/Master-Thesis-SITS/Modeling/Results'
predictions_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/GITHUB/Master-Thesis-SITS/Modeling/Results/predictions.json'

### Model paths for Experiments ###
ae_3d_mvi_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/ae_mvi.pkl'
kmeans_ae_3D_mvi_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/kmeans_ae_mvi.pkl'
ae_3d_b4_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/ae_b4.pkl'
kmeans_ae_3D_b4_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/kmeans_ae_b4.pkl'
best_model_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/3D_AE_temporal_addition_best_model.pkl'
best_8_model_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/3D_AE_8_best_model.pkl'
best_16_model_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/3D_AE_16_best_model.pkl'
best_model_2024_path = '/Users/bhumikasadbhave007/Desktop/THESIS/SITS-models/experiments/3D_AE_temporal_addition_2024_best_model.pkl'

## 2024 data sentinel-2 paths ##
# sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/2024_data/2024_data_train'
# sentinel_base_path_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/2024_data/2024_data_eval'
# save_directory_temporal_train = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/2024_data/patches/train'
# save_directory_temporal_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/2024_data/patches/eval'

## 2024 data extracted sugarbeets paths ##
# load_directory_temporal_train = '/home/k64835/SITS-images-2024/train'
# load_directory_temporal_eval = '/home/k64835/SITS-images-2024/eval'
# deliverable_images_save_path = '/home/k64835/SITS-images-2024/output/train/'
# labels_path = '/home/k64835/SITS-images-2024/labels_2024.csv' 
# sugarbeet_content_csv_path = "/home/k64835/SITS-csv/2024_sugar_content.csv"