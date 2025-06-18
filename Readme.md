# 📊 MLProject

Welcome to **MLProject**!
This project implements multiple clustering algorithms to explore and analyze various synthetic and real-world datasets.

---

## 🚀 Clustering Algorithms Supported

* ⚙️ **K-Means** – Implemented from scratch
* 🧭 **DBSCAN** – Density-Based Spatial Clustering
* 🧿 **Spherical K-Means** – Ideal for cosine-similar data
* 🎲 **Gaussian Mixture Model (GMM)** – Applied to selected datasets

---

## 📁 Available Datasets

* 🌸 **Iris**
* 🔵 **Circles**
* 🌙 **Moons**
* 🎯 **3Gaussians (std = 0.6)**
* 🎯 **3Gaussians (std = 0.9)**

---

## 📈 Features

For each dataset and clustering algorithm, the following are provided:

* 📋 **Descriptive Statistics** – Mean, standard deviation, etc.
* 📊 **Histogram Visualizations** – Distribution of features
* 🔗 **Pair Plots** – Insight into feature relationships
* 📍 **Clustering Visualizations** – Cluster assignment plots

These tools help you understand both data structure and algorithm behavior.

---

## 🛠️ How to Run

You can run the clustering module using the command line. Below is the full list of supported arguments:

### 🔧 CLI Arguments

| Argument         | Type    | Default               | Description                                                  |
| ---------------- | ------- | --------------------- | ------------------------------------------------------------ |
| `--algorithm`    | `str`   | `"kmeans"`            | Clustering algorithm: `kmeans`, `dbscan`, `spherical`, `gmm` |
| `--dataset_name` | `str`   | `"3gaussians-std0.9"` | Dataset to cluster: see available list below                 |
| `--K`            | `int`   | `3`                   | Number of clusters (used in KMeans/Spherical KMeans)         |
| `--iter`         | `int`   | `100`                 | Max number of iterations (for iterative algorithms)          |
| `--seed`         | `int`   | `5`                   | Random seed for reproducibility                              |
| `--tolr`         | `float` | `1e-4`                | Tolerance threshold for convergence (KMeans only)            |
| `--dbscan_eps`   | `float` | `0.5`                 | Epsilon neighborhood for DBSCAN                              |
| `--min_samples`  | `int`   | `5`                   | Minimum number of points for DBSCAN                          |

---

### 💡 Example Usage

```bash
python mains2.py \
  --algorithm kmeans \
  --dataset_name iris \
  --K 3 \
  --iter 50 \
  --seed 42 \
  --tolr 0.0004
```

Another example for DBSCAN:

```bash
python mains2.py \
  --algorithm dbscan \
  --dataset_name moons \
  --dbscan_eps 0.3 \
  --min_samples 4
```

---

### ✅ Supported Dataset Names

* `iris`
* `circles`
* `moons`
* `3gaussians-std0.6`
* `3gaussians-std0.9`

All dataset names are case-insensitive and validated internally.

---

## 📚 Requirements

Install the required dependencies with:


```

---

## 🤝 Contributing

Pull requests are welcome!
Feel free to fork the repository and improve or extend the project.

---

> 🧠 *Machine learning is not just about models — it’s about understanding data.*
> 🎯 *Cluster. Visualize. Learn.*
