from sklearn.cluster import KMeans

def train_kmeans_patches(train_patches, n_clusters, random_state):
    """Function to apply k-means to train_patches 
    (train_patches: can be image tensors or extracted feature tensors)
    """

    flattened_patches = train_patches.reshape(train_patches.size(0), -1).numpy()  
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(flattened_patches)
    return kmeans