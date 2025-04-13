#Path to Raw Sentinel-2 Images and corresponding Sugar-beet Masks (Raw)
sentinel_image_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_images'  
sentinel_mask_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_masks'  
sentinel_id_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_ids'

#temporal raw images path
sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/train/first-10'

#Path to the base directory where image patches of extracted fields/patches are to be saved
save_directory_temporal = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data-Temporal'

#### Local Paths to save extracted patch level images
# base_directory_temporal_train1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-train5'
# base_directory_temporal_test1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-test1'

#### Server Paths ####
load_directory_temporal_train = '/home/k64835/SITS-images/Data-Temporal-train5'
load_directory_temporal_eval = '/home/k64835/SITS-images/Data-Temporal-test1'
images_save_path = '/home/k64835/SITS-images/output/'
labels_path = '/home/k64835/SITS-images/labels.csv'

fields_base_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data'
# labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/test/labels.csv'


#### Trained Model Save Paths: Clustering Algorithms (Baseline 1) ####
kmeans_b10_path = '/home/k64835/SITS-models/baseline_kmeans/kmeans_b10.pkl'
kmedoids_path = '/home/k64835/SITS-models/baseline_kmeans/kmedoids_b10.pkl'
agg_path = '/home/k64835/SITS-models/baseline_kmeans/agg_b10.pkl'

#### Trained Model Save Paths: Classical feature extraction (Baseline 2) ####
kmeans_vi_path = '/home/k64835/SITS-models/baseline_cv/kmeans_vi.pkl'
kmeans_hist_path = '/home/k64835/SITS-models/baseline_cv/kmeans_hist.pkl'
kmeans_pca_path = '/home/k64835/SITS-models/baseline_cv/kmeans_hog.pkl'

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


mae_save_dir = '/home/k64835/'

#### Data Parameters ####
temporal_points = [
    ("june", "2019-06-01", "2019-06-30"),
    ("july", "2019-07-01", "2019-07-31"),
    ("august", "2019-08-01", "2019-08-30"),
    ("september", "2019-09-01", "2019-09-15")
]

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

