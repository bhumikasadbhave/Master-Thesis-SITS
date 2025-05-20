from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
# from sklearn_extra.cluster import KMedoids
import numpy as np

def kmeans_function(train_patches, n_clusters, random_state=10):
    """Function to apply k-means to train_patches 
    (train_patches: can be image tensors or extracted feature tensors)
    """
    if isinstance(train_patches, np.ndarray):
        flattened_patches = train_patches.reshape(train_patches.shape[0], -1)
    else:
        flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()  
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(flattened_patches)
    return kmeans


def agg_clustering_function(train_patches, n_clusters, linkage='ward'):
    """Function to apply Agglomerative Clustering to train_patches
    (train_patches: can be image tensors or extracted feature tensors)
    linkage (str): The linkage criterion to use ('ward', 'complete', 'average', 'single')
    """
    flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()  
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    agg_clustering.fit(flattened_patches)
    return agg_clustering


def db_scan_function(train_patches, eps=0.5, min_samples=5):
    """Function to apply DBSCAN to train_patches
    (train_patches: can be image tensors or extracted feature tensors)
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
    """
    flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()  
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(flattened_patches)
    return dbscan


def kmedoids_function(train_patches, n_clusters, random_state=10, metric='euclidean'):
    """Function to apply K-Medoids to train_patches
    (train_patches: can be image tensors or extracted feature tensors)
    metric (str): The distance metric to use ('euclidean', 'manhattan', etc.)
    """
    flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state, metric=metric)
    kmedoids.fit(flattened_patches)
    return kmedoids


