#Path to Raw Sentinel-2 Images and corresponding Sugar-beet Masks
sentinel_image_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_images'  
sentinel_mask_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_masks'  
sentinel_id_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_ids'

#temporal data path
sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/train/first-10'

#Path to the base directory where image patches of extracted fields are to be saved
save_directory_temporal = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data-Temporal'

#### Path to extracted Temporal images -- fields
# base_directory_temporal_train1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-train5'
# base_directory_temporal_test1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-test1'

#### Server Paths
load_directory_temporal_train = '/home/k64835/SITS-images/Data-Temporal-train5'
load_directory_temporal_eval = '/home/k64835/SITS-images/Data-Temporal-test1'
images_save_path = '/home/k64835/SITS-images/output/'
labels_path = '/home/k64835/SITS-images/labels.csv'


fields_base_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data'
# labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/test/labels.csv'


#### Trained Model Save Paths Baseline 1 ####
kmeans_b10_path = '/home/k64835/SITS-models/baseline_kmeans/kmeans_b10.pkl'
kmeans_bvi_path = '/home/k64835/SITS-models/baseline_kmeans/kmeans_bvi.pkl'
kmeans_mvi_path = '/home/k64835/SITS-models/baseline_kmeans/kmeans_mvi.pkl'

#### Trained Model Save Paths Baseline 2 ####
kmeans_ae_b10_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_b10.pkl'
kmeans_ae_bvi_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_bvi.pkl'
kmeans_ae_mvi_path = '/home/k64835/SITS-models/baseline_ae/kmeans_ae_mvi.pkl'

ae_b10_path = '/home/k64835/SITS-models/baseline_ae/ae_b10.pkl'
ae_bvi_path = '/home/k64835/SITS-models/baseline_ae/ae_bvi.pkl'
ae_mvi_path = '/home/k64835/SITS-models/baseline_ae/ae_mvi.pkl'


#### Data Parameters ####

temporal_points = [
    ("june", "2019-06-01", "2019-06-30"),
    ("july", "2019-07-01", "2019-07-31"),
    ("august", "2019-08-01", "2019-08-30"),
    ("september", "2019-09-01", "2019-09-15")
]

patch_field_size = (64, 64)
subpatch_size = 5
patch_to_field_threshold = 0.2
temporal_stack_size = 7

ae_batch_size = 64
ae_train_test_ratio = 0.8

