"""
Fix Notebook 2: Update K-Means clustering to fit on TRAIN only.

This prevents test set data leakage.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("2_unsupervised_analysis.ipynb")

print("="*60)
print("FIXING NOTEBOOK 2: CLUSTERING DATA LEAKAGE")
print("="*60)

# Backup
import shutil
backup_path = NOTEBOOK_PATH.with_suffix('.ipynb.backup_clustering')
shutil.copy(NOTEBOOK_PATH, backup_path)
print(f"\n[OK] Backup created: {backup_path}")

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"[OK] Loaded notebook: {len(nb['cells'])} cells")

# Find the clustering cell (Cell 14)
target_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'kmeans.fit_predict(features_pca_50)' in source:
            target_cell_idx = i
            print(f"[OK] Found clustering cell at index {i}")
            break

if target_cell_idx is None:
    print("[ERROR] Could not find clustering cell")
    exit(1)

# Corrected source code
corrected_source = """# Apply K-Means on PCA-reduced features (more stable than t-SNE)
print("Applying K-Means clustering (K=2)...")
print("\\n⚠️  IMPORTANT: Fitting on TRAIN data only to prevent test set leakage")

# Separate data by split
train_mask = metadata_df['split'] == 'train'
val_mask = metadata_df['split'] == 'val'
test_mask = metadata_df['split'] == 'test'
unlabeled_mask = metadata_df['split'] == 'unlabeled'

print(f"\\nData splits:")
print(f"  - Train: {train_mask.sum()} samples")
print(f"  - Val: {val_mask.sum()} samples")
print(f"  - Test: {test_mask.sum()} samples")
print(f"  - Unlabeled: {unlabeled_mask.sum()} samples")

# Initialize K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=SEED,
    n_init=10,  # Run 10 times with different initializations
    max_iter=300
)

# FIT on TRAIN data ONLY (prevents test set leakage)
print(f"\\nFitting K-Means on TRAIN split only ({train_mask.sum()} samples)...")
kmeans.fit(features_pca_50[train_mask])

print("[OK] K-Means fitted on train data")
print("  - Cluster centroids learned from train split only")
print("  - Test set NOT used in clustering (prevents leakage)")

# APPLY to ALL data (to get cluster labels for everyone)
print(f"\\nApplying K-Means to all {len(features_pca_50)} samples...")
cluster_labels_kmeans = kmeans.predict(features_pca_50)

print(f"\\n[OK] K-Means clustering completed")
print(f"  - Number of clusters: {kmeans.n_clusters}")
print(f"  - Cluster 0: {(cluster_labels_kmeans == 0).sum()} samples")
print(f"  - Cluster 1: {(cluster_labels_kmeans == 1).sum()} samples")
"""

# Update cell source
nb['cells'][target_cell_idx]['source'] = [line + '\n' for line in corrected_source.split('\n')[:-1]] + [corrected_source.split('\n')[-1]]

# Clear outputs (will need re-execution)
nb['cells'][target_cell_idx]['outputs'] = []
nb['cells'][target_cell_idx]['execution_count'] = None

print(f"[OK] Updated cell {target_cell_idx} with corrected clustering code")

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Saved updated notebook")

print("\n" + "="*60)
print("CHANGES SUMMARY")
print("="*60)
print("\nBEFORE:")
print("  kmeans.fit_predict(features_pca_50)  # ALL 1,506 samples")
print("\nAFTER:")
print("  kmeans.fit(features_pca_50[train_mask])  # TRAIN only (59 samples)")
print("  cluster_labels_kmeans = kmeans.predict(features_pca_50)  # Apply to all")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("\n1. Re-run Notebook 2 from Cell 14 onwards")
print("2. This will regenerate weak_labels.csv with no test leakage")
print("3. Then re-run Notebook 3 to get updated results")

print("\n[OK] Fix applied successfully!")
