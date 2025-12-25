import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import os

# Load data
try:
    features_pca_50 = np.load('features/features_pca_50.npy')
    print(f"Loaded features with shape: {features_pca_50.shape}")
except FileNotFoundError:
    print("Error: features/features_pca_50.npy not found.")
    exit(1)

print(f"{'Eps':<6} {'MinPts':<8} {'Clusters':<10} {'Noise %':<10} {'Silhouette':<10}")
print("-" * 50)

# Grid search ranges
eps_values = np.arange(5, 55, 2.5)  # Range based on previous assumption of 50D space
min_samples_values = [5, 10, 20]

best_score = -1
best_params = None

for min_samples in min_samples_values:
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_pca_50)
        
        # Analyze clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        noise_pct = (n_noise / len(labels)) * 100
        
        silhouette = -1
        if n_clusters >= 2 and n_clusters < 10: # Limit max clusters for readability
             # Calculate silhouette only on non-noise points
            mask = labels != -1
            if np.sum(mask) > n_clusters: # Ensure enough points
                silhouette = silhouette_score(features_pca_50[mask], labels[mask])
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_params = (eps, min_samples)

        if n_clusters >= 2:
             print(f"{eps:<6.1f} {min_samples:<8} {n_clusters:<10} {noise_pct:<10.1f} {silhouette:<10.4f}")

if best_params:
    print("-" * 50)
    print(f"Best configuration found:")
    print(f"Eps: {best_params[0]}")
    print(f"Min_samples: {best_params[1]}")
    print(f"Silhouette Score: {best_score:.4f}")
else:
    print("-" * 50)
    print("No configuration found that produced 2 or more clusters (excluding noise).")
