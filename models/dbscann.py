
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
'''
def estimate_eps(X, k=5):
    """
    Helper function to estimate DBSCAN eps using a k-distance plot.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.title(f"K-Distance Plot (k={k})")
    plt.xlabel("Sorted Data Points")
    plt.ylabel(f"Distance to {k}th Nearest Neighbor")
    plt.grid()
    plt.show()
'''
def run_dbscan(X, eps, min_samples):
   
    
    # --- Step 2: Apply DBSCAN ---
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels
