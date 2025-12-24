"""Test different PCA dimensions to find optimal clustering performance."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set seed
SEED = 42
np.random.seed(SEED)

# Load data
print("Loading data...")
FEATURES_DIR = Path('features')
features = np.load(FEATURES_DIR / 'resnet50_features.npy')
labels = np.load(FEATURES_DIR / 'labels.npy')

# Separate labeled data for evaluation
labeled_mask = labels != -1
labels_labeled = labels[labeled_mask]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Test different PCA dimensions
pca_dims = [10, 20, 30, 50, 75, 100, 150, 200]

print("\n" + "="*80)
print("TESTING DIFFERENT PCA DIMENSIONS")
print("="*80)

results = []

for n_components in pca_dims:
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=SEED)
    features_pca = pca.fit_transform(features_scaled)

    # Calculate variance explained
    variance_explained = np.sum(pca.explained_variance_ratio_)

    # Apply K-Means
    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(features_pca)

    # Evaluate on labeled data only
    cluster_labels_labeled = cluster_labels[labeled_mask]

    # Calculate ARI (try both orientations and pick best)
    ari_original = adjusted_rand_score(labels_labeled, cluster_labels_labeled)
    ari_flipped = adjusted_rand_score(labels_labeled, 1 - cluster_labels_labeled)
    ari = max(ari_original, ari_flipped)

    # Calculate Silhouette score
    silhouette = silhouette_score(features_pca, cluster_labels)

    # Store results
    results.append({
        'n_components': n_components,
        'variance_explained': variance_explained,
        'ari': ari,
        'silhouette': silhouette
    })

    print(f"\nPCA Components: {n_components:3d}")
    print(f"  Variance Explained: {variance_explained:.2%}")
    print(f"  ARI Score:          {ari:.4f}")
    print(f"  Silhouette Score:   {silhouette:.4f}")

# Find optimal
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

best_ari_idx = max(range(len(results)), key=lambda i: results[i]['ari'])
best_sil_idx = max(range(len(results)), key=lambda i: results[i]['silhouette'])

print(f"\n🏆 Best ARI Score:")
print(f"   PCA Components: {results[best_ari_idx]['n_components']}")
print(f"   Variance: {results[best_ari_idx]['variance_explained']:.2%}")
print(f"   ARI: {results[best_ari_idx]['ari']:.4f}")

print(f"\n🏆 Best Silhouette Score:")
print(f"   PCA Components: {results[best_sil_idx]['n_components']}")
print(f"   Variance: {results[best_sil_idx]['variance_explained']:.2%}")
print(f"   Silhouette: {results[best_sil_idx]['silhouette']:.4f}")

print(f"\n📊 Current choice (50 components):")
current_idx = next(i for i, r in enumerate(results) if r['n_components'] == 50)
print(f"   Variance: {results[current_idx]['variance_explained']:.2%}")
print(f"   ARI: {results[current_idx]['ari']:.4f}")
print(f"   Silhouette: {results[current_idx]['silhouette']:.4f}")
print(f"   Rank (by ARI): {sorted(results, key=lambda x: x['ari'], reverse=True).index(results[current_idx]) + 1}/{len(results)}")

print("\n💡 Conclusion:")
if best_ari_idx == current_idx:
    print("   Your current choice (50 components) is OPTIMAL!")
elif results[best_ari_idx]['ari'] - results[current_idx]['ari'] < 0.02:
    print(f"   Your current choice is close to optimal (within {abs(results[best_ari_idx]['ari'] - results[current_idx]['ari']):.4f})")
else:
    print(f"   You could improve ARI by {results[best_ari_idx]['ari'] - results[current_idx]['ari']:.4f}")
    print(f"   by using {results[best_ari_idx]['n_components']} components instead of 50")

# Save results
import json
with open('pca_dimension_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to: pca_dimension_analysis.json")
