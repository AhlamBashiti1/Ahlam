from sklearn.cluster import DBSCAN
from spherecluster import SphericalKMeans

def run_spherical_kmeans_clustering(data, num_clusters, epochs=50):
    
    spherical_model = SphericalKMeans(n_clusters=num_clusters, max_iter=epochs)
    labels = spherical_model.fit_predict(data)
    return labels
