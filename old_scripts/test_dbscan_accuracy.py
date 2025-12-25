import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, accuracy_score

# Load data
try:
    features_pca_50 = np.load('features/features_pca_50.npy')
    labels = np.load('features/labels.npy') # These contain -1 for unlabeled, 0/1 for labeled
    print(f"Loaded features: {features_pca_50.shape}")
    print(f"Loaded labels: {labels.shape}")
except FileNotFoundError:
    print("Error: features/features_pca_50.npy or features/labels.npy not found.")
    exit(1)

# Apply DBSCAN with the "best" parameters found
eps = 37.5
min_samples = 5
print(f"\nApplying DBSCAN (eps={eps}, min_samples={min_samples})...")

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(features_pca_50)

# 1. Filter for labeled data only (exclude -1 in 'labels')
labeled_mask = labels != -1
true_labels = labels[labeled_mask]
pred_labels = cluster_labels[labeled_mask]

# 2. Filter for points that are labeled AND not noise in DBSCAN
valid_mask = pred_labels != -1
true_labels_valid = true_labels[valid_mask]
pred_labels_valid = pred_labels[valid_mask]

# Statistics
n_total_labeled = len(true_labels)
n_noise_labeled = np.sum(pred_labels == -1)
percent_noise = (n_noise_labeled / n_total_labeled) * 100

print(f"\nEvaluation on {n_total_labeled} Expert Labeled Images:")
print(f"--------------------------------------------------")
print(f"Labeled samples discarded as noise: {n_noise_labeled} ({percent_noise:.1f}%)")
print(f"Labeled samples assigned to clusters: {len(true_labels_valid)}")

if len(true_labels_valid) > 0:
    # Calculate ARI
    ari = adjusted_rand_score(true_labels_valid, pred_labels_valid)
    print(f"ARI (on valid points): {ari:.4f}")

    # Calculate Accuracy (requires mapping clusters to labels)
    # We check which cluster majority maps to which label
    # Since we have binary labels (0/1), we test both permutations
    
    # Permutation 1: Cluster 0 -> Label 0, Cluster 1 -> Label 1
    # Note: DBSCAN might output cluster IDs like 0, 1. We map them directly first.
    # If DBSCAN outputs arbitrary IDs (like 0 and 2), we need to be careful.
    unique_clusters = np.unique(pred_labels_valid)
    print(f"Clusters found in labeled data: {unique_clusters}")
    
    if len(unique_clusters) == 2:
        # Simple mapping strategy for 2 clusters
        c1, c2 = unique_clusters
        
        # Option A: c1->0, c2->1
        map_a = np.copy(pred_labels_valid)
        map_a[pred_labels_valid == c1] = 0
        map_a[pred_labels_valid == c2] = 1
        acc_a = accuracy_score(true_labels_valid, map_a)
        
        # Option B: c1->1, c2->0
        map_b = np.copy(pred_labels_valid)
        map_b[pred_labels_valid == c1] = 1
        map_b[pred_labels_valid == c2] = 0
        acc_b = accuracy_score(true_labels_valid, map_b)
        
        best_acc = max(acc_a, acc_b)
        print(f"Accuracy (matching heuristic): {best_acc:.2%}")
    else:
         print(f"Accuracy calculation skipped: Found {len(unique_clusters)} clusters in labeled set (expected 2).")

else:
    print("Cannot calculate metrics: All labeled points were classified as noise.")
