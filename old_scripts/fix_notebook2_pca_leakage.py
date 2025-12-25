"""
Fix Notebook 2: PCA Leakage + Silhouette-based Confidence
- Fit PCA only on TRAIN features (no leakage)
- Transform val/test features separately
- Use silhouette scores for better confidence measurement
"""
import json
import re

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that loads features and fix it to load split-specific files
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        # Fix feature loading to use split-specific files
        if 'features = np.load(FEATURES_DIR' in source and 'resnet50_features.npy' in source:
            new_source = """# Load SPLIT-SPECIFIC features (prevents PCA leakage)
train_features = np.load(FEATURES_DIR / 'train_features.npy')
train_labels = np.load(FEATURES_DIR / 'train_labels.npy')
val_features = np.load(FEATURES_DIR / 'val_features.npy')
val_labels = np.load(FEATURES_DIR / 'val_labels.npy')
test_features = np.load(FEATURES_DIR / 'test_features.npy')
test_labels = np.load(FEATURES_DIR / 'test_labels.npy')
unlabeled_features = np.load(FEATURES_DIR / 'unlabeled_features.npy')
unlabeled_labels = np.load(FEATURES_DIR / 'unlabeled_labels.npy')

# Combine labeled features for evaluation (train+val+test)
labeled_features = np.vstack([train_features, val_features, test_features])
labeled_labels = np.hstack([train_labels, val_labels, test_labels])

# All features (for clustering)
features = np.vstack([labeled_features, unlabeled_features])
labels = np.hstack([labeled_labels, unlabeled_labels])

# Load metadata
metadata_df = pd.read_csv(FEATURES_DIR / 'metadata.csv')

# Add label_name if missing
if 'label_name' not in metadata_df.columns:
    metadata_df['label_name'] = pd.Series(labels).map({0: 'normal', 1: 'cancer', -1: 'unlabeled'})

print("="*60)
print("DATA LOADED SUCCESSFULLY - SPLIT-SPECIFIC")
print("="*60)
print(f"\\nTrain features: {train_features.shape}")
print(f"Val features: {val_features.shape}")
print(f"Test features: {test_features.shape}")
print(f"Unlabeled features: {unlabeled_features.shape}")
print(f"\\nCombined features: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Metadata rows: {len(metadata_df)}")

# Separate labeled and unlabeled data
labeled_mask = labels != -1
unlabeled_mask = labels == -1

features_labeled = features[labeled_mask]
labels_labeled = labels[labeled_mask]
features_unlabeled = features[unlabeled_mask]

print(f"\\n📊 Data Split:")
print(f"  - Labeled samples: {features_labeled.shape[0]}")
print(f"    • Normal (0): {(labels_labeled == 0).sum()}")
print(f"    • Cancer (1): {(labels_labeled == 1).sum()}")
print(f"  - Unlabeled samples: {features_unlabeled.shape[0]}")
print(f"  - Feature dimensions: {features.shape[1]}")
"""
            cell['source'] = new_source.split('\n')
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            print(f"Fixed cell {i}: Load split-specific features")

        # Fix PCA fitting to use only TRAIN features
        if 'pca_50 = PCA(n_components=50' in source and 'fit_transform(features_scaled)' in source:
            new_source = """# Apply PCA - FIT ONLY ON TRAIN (prevents leakage)
print("Applying PCA - fit on TRAIN, transform on others...")

# Standardize features FIRST (fit on train only)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
val_features_scaled = scaler.transform(val_features)
test_features_scaled = scaler.transform(test_features)
unlabeled_features_scaled = scaler.transform(unlabeled_features)

# Combine for clustering (all scaled with same scaler fitted on train)
labeled_features_scaled = np.vstack([train_features_scaled, val_features_scaled, test_features_scaled])
features_scaled = np.vstack([labeled_features_scaled, unlabeled_features_scaled])

print(f"\\nFeatures standardized (fitted on train only)")

# FIT PCA only on TRAIN features (prevents leakage)
pca_50 = PCA(n_components=50, random_state=SEED)
train_features_pca = pca_50.fit_transform(train_features_scaled)

# TRANSFORM val/test/unlabeled with fitted PCA
val_features_pca = pca_50.transform(val_features_scaled)
test_features_pca = pca_50.transform(test_features_scaled)
unlabeled_features_pca = pca_50.transform(unlabeled_features_scaled)

# Combine for clustering
labeled_features_pca = np.vstack([train_features_pca, val_features_pca, test_features_pca])
features_pca_50 = np.vstack([labeled_features_pca, unlabeled_features_pca])

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_50.explained_variance_ratio_)

print(f"\\n✓ PCA completed (NO LEAKAGE)")
print(f"  - Fitted on: TRAIN only ({train_features.shape[0]} samples)")
print(f"  - Transformed: Val ({val_features.shape[0]}), Test ({test_features.shape[0]}), Unlabeled ({unlabeled_features.shape[0]})")
print(f"  - Original dimensions: {features_scaled.shape[1]}")
print(f"  - Reduced dimensions: {features_pca_50.shape[1]}")
print(f"  - Variance explained: {cumulative_variance[-1]:.2%}")
print(f"  - Train PCA shape: {train_features_pca.shape}")
print(f"  - Combined PCA shape: {features_pca_50.shape}")
"""
            cell['source'] = new_source.split('\n')
            cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            print(f"Fixed cell {i}: PCA fit only on train")

# Fix confidence calculation to use silhouette scores
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])

        if 'confidence_scores = 1 - (dist_to_assigned / dist_to_other)' in source:
            # Replace distance-based confidence with silhouette-based
            new_confidence_calc = """# Calculate confidence using SILHOUETTE SCORES (better than distance-based)
from sklearn.metrics import silhouette_samples

print("Calculating confidence using silhouette scores...")

# Silhouette score per sample: measures how similar a sample is to its cluster vs other cluster
# Range: [-1, 1], higher is better
# > 0.5: Strong assignment, 0.3-0.5: Moderate, < 0.3: Weak/ambiguous
silhouette_scores_all = silhouette_samples(features_pca_50, aligned_clusters)

# Normalize to [0, 1] range for consistency
# Original range [-1, 1] -> [0, 1]
confidence_scores = (silhouette_scores_all + 1) / 2

print(f"\\nSilhouette-based Confidence Statistics:")
print(f"  - Mean:   {confidence_scores.mean():.4f}")
print(f"  - Median: {np.median(confidence_scores):.4f}")
print(f"  - Std:    {confidence_scores.std():.4f}")
print(f"  - Min:    {confidence_scores.min():.4f}")
print(f"  - Max:    {confidence_scores.max():.4f}")
print(f"  - 25th percentile: {np.percentile(confidence_scores, 25):.4f}")
print(f"  - 50th percentile: {np.percentile(confidence_scores, 50):.4f}")
print(f"  - 75th percentile: {np.percentile(confidence_scores, 75):.4f}")
print(f"  - 80th percentile: {np.percentile(confidence_scores, 80):.4f}")
print(f"  - 90th percentile: {np.percentile(confidence_scores, 90):.4f}")

# OLD distance-based method (kept for comparison):
# distances = cdist(features_pca_50, centroids, metric='euclidean')
# dist_to_assigned = np.array([distances[i, aligned_clusters[i]] for i in range(len(aligned_clusters))])
# dist_to_other = np.array([distances[i, 1-aligned_clusters[i]] for i in range(len(aligned_clusters))])
# confidence_scores_old = 1 - (dist_to_assigned / dist_to_other)
"""
            # Find the exact location and replace
            source_lines = cell['source']
            for j, line in enumerate(source_lines):
                if 'confidence_scores = 1 - (dist_to_assigned / dist_to_other)' in line:
                    # Find the start of the confidence calculation block
                    start_idx = j
                    while start_idx > 0 and 'Calculate confidence' not in source_lines[start_idx-1]:
                        start_idx -= 1
                    if start_idx > 0:
                        start_idx -= 1

                    # Replace from distance calculation to confidence calculation
                    # Keep everything before, replace the calculation, keep everything after
                    new_lines = source_lines[:start_idx] + new_confidence_calc.split('\n')
                    new_lines = [l + '\n' for l in new_lines[:-1]] + [new_lines[-1]]

                    # Find where the confidence block ends (next print or empty line)
                    end_idx = j + 1
                    while end_idx < len(source_lines) and source_lines[end_idx].strip() and not source_lines[end_idx].strip().startswith('# Apply'):
                        end_idx += 1

                    # Combine: before + new calc + after
                    cell['source'] = new_lines + source_lines[end_idx:]
                    print(f"Fixed cell {i}: Silhouette-based confidence scoring")
                    break

# Save updated notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\n" + "="*60)
print("NOTEBOOK 2 FIXES APPLIED:")
print("="*60)
print("1. Load split-specific features (train/val/test separate)")
print("2. Fit PCA only on TRAIN (no leakage)")
print("3. Transform val/test/unlabeled with fitted PCA")
print("4. Silhouette-based confidence (better than distance)")
print("\nNotebook 2 ready for execution!")
