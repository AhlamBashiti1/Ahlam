import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score

# =====================
# Data Loading
# =====================
def load_dataset(name):
    if name == "iris":
        iris = load_iris()
        X = iris.data
        y = iris.target
    else:
        path = f"Data/{name}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Dataset file not found: '{path}'. Please make sure the file exists.")
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    return X, y

# =====================
# PCA Utility
# =====================
def apply_pca(X, n_components=2):
    if X.shape[1] > n_components:
        print("[INFO] PCA applied for visualization.")
        return PCA(n_components=n_components).fit_transform(X)
    print("[INFO] PCA not needed.")
    return X

# =====================
# Cluster Plotting (KMeans, DBSCAN, etc.)
# =====================
def plot_clusters(X, labels, title="Clustering", centroids=None):
    X_pca = apply_pca(X, 2)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("Set1", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_points = X_pca[cluster_mask]
        color = color_map[label]
        marker = "x" if label == -1 else "o"
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=[color], label=label_name, marker=marker,
            edgecolor='k', alpha=0.8, s=60
        )

    if centroids is not None:
        centroids_pca = apply_pca(centroids, 2)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    c='black', marker='x', s=200, linewidths=2, label='Centroids')

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()



def plot_clusters_with_palette(X, labels, title, centroids=None):
    """
    Plot 2D clusters using a Seaborn color palette and optional centroids.
    Saves the plot in the 'plots/' directory.
    """
    # Ensure plots/ directory exists
    os.makedirs("images", exist_ok=True)
    
    # Apply PCA to reduce X to 2D
    X_pca = apply_pca(X, 2)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("Set1", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_points = X_pca[cluster_mask]
        color = color_map[label]
        marker = "x" if label == -1 else "o"
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1],
            c=[color], label=label_name, marker=marker,
            edgecolor='k', alpha=0.8, s=60
        )
    
    if centroids is not None:
        centroids_pca = apply_pca(centroids, 2)
        plt.scatter(
            centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', marker='X', s=200, linewidths=2, label='Centroids'
        )
    
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    filename = f"images/{title.replace(' ', '_').replace('-', '').lower()}.png"
    plt.savefig(filename)
    plt.close()    

# =====================
# Cluster Comparison
# =====================
def plot_original_and_predicted_clusters(X, y_original, y_pred, centroids=None, title="Original_vs_Predicted"):

    X_pca = apply_pca(X, 2)
    centroids_pca = apply_pca(centroids, 2) if centroids is not None else None
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for label in np.unique(y_original):
        axes[0].scatter(X_pca[y_original == label, 0], X_pca[y_original == label, 1],
                        color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')
    axes[0].set_title('Original Clusters')
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].legend()
    axes[0].grid(True)

    for label in np.unique(y_pred):
        axes[1].scatter(X_pca[y_pred == label, 0], X_pca[y_pred == label, 1],
                        color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')
    if centroids_pca is not None:
        axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                        color='black', marker='x', s=200, label='Centroids')
    axes[1].set_title('Predicted Clusters')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()


# =====================
# K-distance Graph (for DBSCAN)
# =====================
def plot_k_distance(X, k=4):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.title(f"K-distance Graph (k={k})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}th Nearest Neighbor Distance")
    plt.grid(True)
    plt.savefig(f"images/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()

    print("Suggestion: Pick eps around the elbow (you are seeing the curve).")

# =====================
# Cluster Summary Print
# =====================
def print_cluster_summary(labels):
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    total_points = len(labels)

    print("="*40)
    print("Cluster Summary")
    print("="*40)
    print(f"Total data points     : {total_points}")
    print(f"Number of clusters    : {n_clusters}")
    print(f"Noise points (-1)     : {n_noise} ({(n_noise / total_points) * 100:.2f}%)")

    if n_clusters > 0:
        cluster_counts = Counter(labels[labels != -1])
        print("Cluster sizes:")
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"   - Cluster {cluster_id}: {count} points")
    print("="*40)

# =====================
# Cluster Comparison
# =====================
def  Old_Predecited(X, y_original, y_pred, centroids, title="Old_vs_Predicted"):
    # Apply PCA to reduce the dataset to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    # Define colors
    colors = ['red', 'blue', 'green'] 
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original clusters
    for label in np.unique(y_original):
            axes[0].scatter(X_pca[y_original == label, 0], X_pca[y_original == label, 1], 
                            color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')

    axes[0].set_title('Original Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot predicted clusters and centroids
    for label in np.unique(y_pred):
        axes[1].scatter(X_pca[y_pred == label, 0], X_pca[y_pred == label, 1], 
                        color=colors[int(label) % len(colors)], label=f'Cluster {int(label)}', edgecolor='k')
    
    # Plot centroids
    axes[1].scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    color='black', marker='x', s=200, label='Centroids')
    axes[1].set_title('Predicted Clusters')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()


# =====================
# Evaluation Metrics Plot
# =====================
def plot_evaluation_metrics(losses, j_value, K, title="Cost"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o', color='blue', markersize=3)
    plt.title(f'K-means Cost (J) for K={K} Clusters ')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (J)')
    plt.axhline(y=j_value, color='red', linestyle='--', label=f'Final J: {j_value:.4f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/{title.replace(' ', '_').replace('-', '').lower()}.png")
    plt.close()

