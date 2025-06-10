#Path to Raw Sentinel-2 Images and corresponding Sugar-beet Masks (Raw)
sentinel_image_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_images'  
sentinel_mask_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_masks'  
sentinel_id_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_ids'


### --- Paths for field extraction and saving from 1000x1000 images --- ###

#temporal raw images path
sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/train/'
sentinel_base_path_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/eval/'

#Path to the base directory where image patches of extracted fields/patches are to be saved
save_directory_temporal_train = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Train'
save_directory_temporal_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Eval'

#### Local Paths to save extracted patch level images
# load_directory_temporal_train = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-train5'
# load_directory_temporal_eval = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-test1'
# labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/test/labels.csv'
# sugarbeet_content_csv_path = "/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/2019_sugar_content.csv"
# fields_base_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data'
# labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/test/labels.csv'

#### --- Server Paths --- ####
load_directory_temporal_train = '/home/k64835/SITS-images/Data-Temporal-train5'
load_directory_temporal_eval = '/home/k64835/SITS-images/Data-Temporal-test1'
deliverable_images_save_path = '/home/k64835/SITS-images/output/train/'
labels_path = '/home/k64835/SITS-images/labels.csv'

#### Filtering fields that are not sugarbeet 
sugarbeet_content_csv_path = "/home/k64835/SITS-csv/2019_sugar_content.csv"

#### Trained Model Save Paths: Clustering Algorithms (Baseline 1) ####
kmeans_b10_path = '/home/k64835/SITS-models/baseline_kmeans/kmeans_b10.pkl'
kmedoids_path = '/home/k64835/SITS-models/baseline_kmeans/kmedoids_b10.pkl'
agg_path = '/home/k64835/SITS-models/baseline_kmeans/agg_b10.pkl'

#### Trained Model Save Paths: Classical feature extraction (Baseline 2) ####
kmeans_vi_path = '/home/k64835/SITS-models/baseline_cv/kmeans_vi.pkl'
kmeans_hist_path = '/home/k64835/SITS-models/baseline_cv/kmeans_hist.pkl'
kmeans_pca_path = '/home/k64835/SITS-models/baseline_cv/kmeans_hog.pkl'

#### Trained Model Save Paths: Pre-trained models feature extraction (Baseline 3) ####
resnet3D_path = '/home/k64835/SITS-models/baseline_pretrained/resnet_3D.pkl'
vit_imagenet_path = '/home/k64835/SITS-models/baseline_pretrained/vit_imagenet.pkl'
resent_sentinel_path = '/home/k64835/SITS-models/baseline_pretrained/resnet_sentinel.pkl'

#### Trained Model Save Paths: Autoencoders main ####
kmeans_ae_3D_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_3D.pkl'
kmeans_ae_2D_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_2D.pkl'
kmeans_vae_3D_path = '/home/k64835/SITS-models/baseline_ae/kmeans_vae_3D.pkl'
kmeans_vae_2D_path = '/home/k64835/SITS-models/baseline_ae/kmeans_vae_2D.pkl'

ae_3d_path = '/home/k64835/SITS-models/baseline_ae/ae_3D.pkl'
ae_2D_path = '/home/k64835/SITS-models/baseline_ae/ae_2D.pkl'
vae_3D_path = '/home/k64835/SITS-models/baseline_ae/vae_3D.pkl'
vae_2D_path = '/home/k64835/SITS-models/baseline_ae/vae_2D.pkl'

#### Best performing Autoencoder with Temporal Encodings (Final Model) ####
kmeans_ae_3D_TE_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_3D_TS.pkl'
ae_3d_TE_path = '/home/k64835/SITS-models/baseline_ae/ae_3D_TE.pkl'

kmeans_ae_3D_TEadd_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_3D_TSadd.pkl'
ae_3d_TEadd_path = '/home/k64835/SITS-models/baseline_ae/ae_3D_TEadd.pkl'


### Model paths for Experiments ###
# --- Regression --- #
regression_linear_flat = '/home/k64835/SITS-models/regression/linear_flat.pkl'
regression_linear_hist = '/home/k64835/SITS-models/regression/linear_hist.pkl'
regression_linear_2dconv = '/home/k64835/SITS-models/regression/linear_2dconv.pkl'
regression_linear_3dconv = '/home/k64835/SITS-models/regression/linear_3dconv.pkl'

# -- Vegetation Index Experiments -- #
ae_3d_mvi_path = '/home/k64835/SITS-models/vi/ae_mvi.pkl'
kmeans_ae_3D_mvi_path = '/home/k64835/SITS-models/vi/kmeans_ae_mvi.pkl'
ae_3d_b4_path = '/home/k64835/SITS-models/vi/ae_b4.pkl'
kmeans_ae_3D_b4_path = '/home/k64835/SITS-models/vi/kmeans_ae_b4.pkl'


# -- Result JSONs for all AE-based models and experiments -- ##
results_json_path = '/home/k64835/Master-Thesis-SITS/Modeling/Results'
predictions_path = '/home/k64835/Master-Thesis-SITS/Modeling/Results/predictions.json'

# -- Best Model Saved Model path -- #
best_model_path = '/home/k64835/Master-Thesis-SITS/Modeling/Results/Trained_Models/3D_AE_temporal_addition_best_model.pkl'
best_8_model_path = '/home/k64835/Master-Thesis-SITS/Modeling/Results/Trained_Models/3D_AE_8_best_model.pkl'
best_16_model_path = '/home/k64835/Master-Thesis-SITS/Modeling/Results/Trained_Models/3D_AE_16_best_model.pkl'


#### Data Parameters ####
temporal_points = [
    ("june", "2019-06-01", "2019-06-30"),
    ("july", "2019-07-01", "2019-07-31"),
    ("august", "2019-08-01", "2019-08-30"),
    ("september", "2019-09-01", "2019-09-15")
]
reference_date_temp_encoding='20190601.0'


#### --- Local Paths --- ####
# kmeans_b10_local_path = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_kmeans/kmeans_b10.pkl'
# kmeans_hist_local_path = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_cv/kmeans_hist.pkl'
# ae_3d_local_path = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_ae/ae_3D_TEadd.pkl'
# ae_kmeans_3d_local = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_ae/kmeans_ae_3D_TSadd.pkl'
# ae_2D_local_path = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_ae/ae_2D.pkl'
# ae_kmeans_2d_local = '/Users/bhumikasadbhave007/Desktop/Thesis_Models/SITS-models/baseline_ae/kmeans_ae_2D.pkl'
# predictions_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/GITHUB/Master-Thesis-SITS/Modeling/Results/predictions.json'

#### Other Parameters ####
patch_size = (64, 64)
subpatch_size = 4
batch_size=64
subpatch_to_patch_threshold = 0.5
temporal_stack_size = 7
num_encoding_channels = 2
ref_date = '20190601.0'
max_date_diff = 115

ae_batch_size = 64
ae_train_test_ratio = 0.8
mae_save_dir = '/home/k64835/'
