#Path to Raw Sentinel-2 Images and corresponding Sugar-beet Masks
sentinel_image_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_images'  
sentinel_mask_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_masks'  
sentinel_id_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Data-Sample/1100x1100/sentinel2_ids'

#temporal data path
sentinel_base_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/train/first-10'

#Size of individual patches/fields - all extracted sugar-beet fields will be scaled to this size
field_size = (64, 64)

#Path to the base directory where image patches of extracted fields are to be saved
base_directory_temporal = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data-Temporal'

#### Path to extracted Temporal images -- fields
# base_directory_temporal_train1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-train5'
# base_directory_temporal_test1 = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/Data-Temporal-test1'

#### Server Paths
base_directory_temporal_train1 = '/home/k64835/SITS-images/Data-Temporal-train5'
base_directory_temporal_test1 = '/home/k64835/SITS-images/Data-Temporal-test1'
save_path = '/home/k64835/SITS-images/output/'
labels_path = '/home/k64835/SITS-images/labels.csv'


fields_base_directory = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Data-Preprocessing/Data'
# labels_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Temporal-Data/test/labels.csv'
trained_models_path = '/Users/bhumikasadbhave007/Documents/THWS/Semester-4/MASTER-THESIS/Master-Thesis-Github/Master-Thesis/Modeling/Trained_Models/'

patch_to_field_threshold = 0.2

temporal_stack_size = 7
# temporal_points_old = [
#     ("early_june", "2019-06-01", "2019-06-10"),
#     ("mid_june", "2019-06-11", "2019-06-20"),
#     ("late_june", "2019-06-21", "2019-06-30"),
#     ("early_july", "2019-07-01", "2019-07-10"),
#     ("mid_july", "2019-07-11", "2019-07-20"),
#     ("late_july", "2019-07-21", "2019-07-31"),
#     ("early_august", "2019-08-01", "2019-08-10"),
#     ("mid_august", "2019-08-11", "2019-08-20"),
#     ("late_august", "2019-08-21", "2019-08-31"),
#     ("early_september", "2019-09-01", "2019-09-15")
# ]

temporal_points = [
    ("june", "2019-06-01", "2019-06-30"),
    ("july", "2019-07-01", "2019-07-31"),
    ("august", "2019-08-01", "2019-08-30"),
    ("september", "2019-09-01", "2019-09-15")
]

# temporal_points = [
#     ("early_june", "2019-06-01", "2019-06-15"),
#     ("late_june", "2019-06-16", "2019-06-30"),
#     ("early_july", "2019-07-01", "2019-07-15"),
#     ("late_july", "2019-07-16", "2019-07-31"),
#     ("early_august", "2019-08-01", "2019-08-15"),
#     ("late_august", "2019-08-16", "2019-08-31"),
#     ("early_september", "2019-09-01", "2019-09-15")
# ]



temporal_interpolation = True

# 'ndvi' , 'mcai' , 'ari' , 'ndvi-mcai' or 'all'
vegetation_index = 'ndvi'

#argpass
#seeding