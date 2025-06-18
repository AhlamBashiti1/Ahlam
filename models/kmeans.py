from scipy.spatial.distance import pdist
import itertools
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class KMeans:
    def __init__(self, K=3, max_iters=100, tol=1e-4, seed=None):
        """
        KMeans clustering algorithm.

        Parameters:
        -----------
        K : int Number of clusters
        max_iters : int Maximum iterations to run
        tol : float Convergence tolerance (centroid movement threshold)
        seed : int or None Random seed for reproducible initialization
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed

        self.centroids = None
        self.j = float('inf')
        self.losses = []
        
        
        if self.seed is not None:
            np.random.seed(self.seed)
    # select random points centroids        
    def initialize_selected_clusters(self, data): 
        n_samples, _ = data.shape
        centroids_indices = np.random.choice(n_samples, self.K, replace=False) 
        self.centroids = np.asarray(data[centroids_indices])
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        indices = np.random.choice(data.shape[0], self.K, replace=False)
       '''
   
    def initialize_clusters(self, data): # random initialization
        _, n_features = data.shape
        self.centroids = np.random.rand(self.K, n_features) * (data.max() - data.min()) + data.min()

    def initialize_distant_centroids(self, data): # distant points
        c = [list(x) for x in itertools.combinations(range(len(data)), self.K)]
        distances = []
        for i in c:    
            distances.append(np.mean(pdist(data[i, :])))
        
        ind = distances.index(max(distances))
        rows = c[ind]
        self.centroids = data[rows]

    def find_nearest_centroid(self, data):
        
        nearest_centroids = np.zeros(len(data), dtype=int)
        
        for i, point in enumerate(data):
            min_distance = float('inf')
            nearest_centroid_idx = -1
            
            for j, centroid in enumerate(self.centroids):
                distance = np.sqrt(np.sum((point - centroid) ** 2))
                if distance < min_distance:
                    min_distance = distance
                    nearest_centroid_idx = j
            
            nearest_centroids[i] = nearest_centroid_idx
        
        return nearest_centroids
        ''' 
        # Vectorized computation of squared Euclidean distances
        distances = np.sum((data[:, np.newaxis] - self.centroids) ** 2, axis=2)
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)  
        '''
    def assign_labels(self, X):
        #Compute squared Euclidean distances between each point and each centroid
        #distances = np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)
   
    def update_centroids(self, data, nearest_centroids):
        centroids = np.zeros((self.K, data.shape[1]))
        for cluster_idx in range(self.K):
            cluster_points = data[nearest_centroids == cluster_idx]
            if len(cluster_points) > 0:
                centroids[cluster_idx] = np.mean(cluster_points, axis=0)
            else:
                centroids[cluster_idx] = self.centroids[cluster_idx]
        self.centroids = centroids  

    def objective_function(self, data, nearest_centroids):
        '''
        total_distance = 0
        for k in range(self.K):
            mask = nearest_centroids == k
            if np.any(mask):
                diff = data[mask] - self.centroids[k]
            total_distance += np.sum(np.square(diff))
        '''
        total_distance = 0
        for cluster_idx, centroid in enumerate(self.centroids):
            cluster_points = data[nearest_centroids == cluster_idx]
            #total_distance += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
            diff = cluster_points - centroid
            total_distance += np.sum(np.einsum('ij,ij->i', diff, diff))

        self.j = total_distance
        self.losses.append(total_distance)

    def evaluation(self, data, nearest_centroids):
       # silhouette = silhouette_score(data, nearest_centroids) # higher is better (ranging from -1 to 1)
       # davies_bouldin = davies_bouldin_score(data, nearest_centroids) # lower is better
       # self.silhouettes.append(silhouette)
       # self.davies_bouldins.append(davies_bouldin)
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  ➤ Number of clusters: {self.K}")
        logger.info(f"  ➤ Objective function (J): {self.j:.4f}")
        logger.info(f"  ➤ Current Loss: {self.losses[-1]:.4f}")
        #logger.info(f"  ➤ Silhouette Score: {silhouette:.4f}")
        #logger.info(f"  ➤ Davies-Bouldin Index: {davies_bouldin:.4f}")

        # Optional: Use debug for verbose outputs
        logger.debug(f"  ➤ Centroids: \n{self.centroids}")
        logger.debug(f"  ➤ All Losses: {self.losses}\n")

    def fit(self, data):
        # Initialize centroids
        self.initialize_distant_centroids(data)
        
        # Assign initial labels
        nearest_centroids = self.assign_labels(data)
        
        # Compute initial objective function and evaluation
        self.objective_function(data, nearest_centroids)
        self.evaluation(data, nearest_centroids)
        
        for i in range(self.max_iters):
            logger.info(f"Iteration {i + 1}")

            prev_j = self.j

            # Assign labels and update centroids
            nearest_centroids = self.assign_labels(data)
            self.update_centroids(data, nearest_centroids)

            # Update loss and evaluation
            self.objective_function(data, nearest_centroids)

            # Check for convergence early before computing evaluation (saves time)
            if abs(prev_j - self.j) < self.tol:
                logger.info(f"Converged at iteration {i + 1}. ΔJ = {abs(prev_j - self.j):.6f}")
                self.evaluation(data, nearest_centroids)
                break

            self.evaluation(data, nearest_centroids)
       
        return self.centroids, nearest_centroids, self.j