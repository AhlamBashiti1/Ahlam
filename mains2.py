'''
main.py - Entry point for clustering experiments.
Supports KMeans, DBSCAN, and (future) Spherical KMeans on synthetic and real datasets.
'''
 
# -----------------------------
# Imports
# -----------------------------
# Standard Library
import argparse
import logging
 
# Third-party Libraries
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
 
# Local Modules
from models.kmeans import KMeans
from models.dbscann import run_dbscan
from spherical import run_spherical_kmeans_clustering # type: ignore
from utils2 import plot_clusters_with_palette
from utils2 import plot_original_and_predicted_clusters
from utils2 import Old_Predecited


from utils2 import (
    load_dataset,
    plot_clusters,
    plot_clusters_with_palette,
    plot_evaluation_metrics,
    plot_original_and_predicted_clusters,
    Old_Predecited,
    plot_k_distance,
    print_cluster_summary
)
 
# -----------------------------
# Configurations
# -----------------------------
CONFIG = {
    "CLUSTERING_ALGORITHMS": ["kmeans", "dbscan", "spherical"],
    "DATASETS": ["moons", "circles", "iris", "3gaussians-std0.6", "3gaussians-std0.9"],
    "DBSCAN_PARAMS": {
        "iris": {"eps": 0.5, "min_samples": 5},
        "moons": {"eps": 0.3, "min_samples": 5},
        "circles": {"eps": 0.2, "min_samples": 4},
        "3gaussians-std0.6": {"eps": 0.4, "min_samples": 10},
        "3gaussians-std0.9": {"eps": 0.15473, "min_samples": 5},
    }
}
 
# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# -----------------------------
# KMeans Runner
# -----------------------------
def run_kmeans(X, y, args):
    logger.info(f"Running KMeans on {args.dataset_name}...")
    kmeans = KMeans(K=args.K, max_iters=args.iter, tol=args.tolr, seed=args.seed)
    centroids, nearest_centroids, error = kmeans.fit(X)
    Old_Predecited(X, y, nearest_centroids, centroids)
    plot_evaluation_metrics(kmeans.losses, kmeans.j, args.K)
 
# -----------------------------
# DBSCAN Runner
# -----------------------------
def process_dbscan(X, args):
    dataset_name = args.dataset_name
 
    if dataset_name in CONFIG["DBSCAN_PARAMS"]:
        eps = CONFIG["DBSCAN_PARAMS"][dataset_name]["eps"]
        min_samples = CONFIG["DBSCAN_PARAMS"][dataset_name]["min_samples"]
    else:
        eps = args.dbscan_eps
        min_samples = args.min_samples
 
    logger.info(f"Running DBSCAN on {dataset_name} with eps={eps}, min_samples={min_samples}...")
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    plot_k_distance(X_scaled, k=4)
    labels = run_dbscan(X_scaled, eps=eps, min_samples=min_samples)
    plot_clusters(X_scaled, labels, title=f"DBSCAN Clustering {dataset_name.capitalize()}")
    print_cluster_summary(labels)
 
    if dataset_name == "3gaussians-std0.9":
        logger.info("Running GMM on 3gaussians-std0.9...")
        gmm = GaussianMixture(n_components=args.K, random_state=args.seed)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        plot_clusters(X_scaled, labels, title=f"GMM Clustering {dataset_name.capitalize()}")
        print_cluster_summary(labels)

# -----------------------------
# Spherical KMeans Runner
# -----------------------------
def run_spherical_kmeans(X, y, args):
    dataset_name = args.dataset_name
    logger.info(f"Running Spherical KMeans on {args.dataset_name}...")
    
    # Optionally, you can scale data (some people recommend it for Spherical KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    labels = run_spherical_kmeans_clustering(X_scaled, num_clusters=args.K, epochs=args.iter)

    plot_original_and_predicted_clusters(X_scaled, y, labels, centroids=None, title=f"Spherical Clustering {dataset_name.capitalize()}")  # centroids not returned by spherecluster
    plot_clusters_with_palette(X_scaled, labels, title=f"Spherical KMeans Clustering {args.dataset_name.capitalize()}")
    print_cluster_summary(labels)

 
# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", type=str, default="kmeans", choices=CONFIG["CLUSTERING_ALGORITHMS"],
        help=f"Clustering algorithm to use: {', '.join(CONFIG['CLUSTERING_ALGORITHMS'])}"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="3gaussians-std0.9", choices=CONFIG["DATASETS"],
        help=f"Dataset name ({', '.join(CONFIG['DATASETS'])})"
    )
    parser.add_argument("--K", type=int, default=3, help="Number of clusters (used in KMeans/Spherical)")
    parser.add_argument("--iter", type=int, default=100, help="Max number of iterations")
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument("--tolr", type=float, default=1e-4, help="Convergence threshold (for KMeans)")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN minimum samples")
 
    args = parser.parse_args()
    args.algorithm = args.algorithm.lower()
    args.dataset_name = args.dataset_name.lower()
 
    if args.dataset_name not in CONFIG["DATASETS"]:
        raise ValueError(f"Unsupported dataset {args.dataset_name}. Choose from: {CONFIG['DATASETS']}")
 
    return args
 
# -----------------------------
# Main Execution
# -----------------------------
def main(args):
    X, y = load_dataset(args.dataset_name)
    logger.info(f"Dataset {args.dataset_name} loaded successfully with shape {X.shape}.")
 
    if args.algorithm == "kmeans":
        run_kmeans(X, y, args)
    elif args.algorithm == "dbscan":
        process_dbscan(X, args)
    elif args.algorithm == "spherical":
        run_spherical_kmeans(X, y, args)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}. Choose from {CONFIG['CLUSTERING_ALGORITHMS']}.")
 
# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as e:
        logger.error("Error occurred:", exc_info=True)
 