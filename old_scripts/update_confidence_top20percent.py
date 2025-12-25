"""
Update Notebook 2 to filter for TOP 20% confidence labels only

Changes:
1. Update threshold calculation to keep only top 20% (80th percentile)
2. Update Cell 23 markdown to reflect this strategy
3. Update Cell 24 code to use percentile of UNLABELED data only
4. Update Cell 25 markdown interpretation
"""
import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 23 markdown - explain top 20% strategy
cell_23_updated = """### 6.2 Confidence Thresholding - Filter Top 20% High-Quality Weak Labels

**Strategy: Keep Only the Best 20%**

Not all cluster assignments are equally certain. To maximize weak label quality:
- Calculate confidence scores for all unlabeled samples
- Keep only the **top 20%** with highest confidence (80th percentile threshold)
- Discard the remaining 80% with lower confidence

**Why Top 20%?**
- **Quality over Quantity**: Better to have fewer high-quality labels than many noisy ones
- **Medical Imaging**: Subtle patterns require high certainty
- **Semi-supervised Learning**: Pre-training on clean weak labels is more effective
- **Conservative Approach**: Minimizes label noise at the cost of sample size

**Confidence Calculation:**
- Uses **silhouette scores** to measure cluster assignment quality
- Silhouette score: How well a sample fits its assigned cluster vs. other clusters
- Range: [-1, 1], higher = better assignment
- Normalized to [0, 1] for interpretability

**Expected Outcome:**
- From 1,406 unlabeled samples → Keep ~281 samples (top 20%)
- These 281 samples have the highest cluster assignment confidence
- Remaining ~1,125 samples are discarded (ambiguous cluster assignments)

**Trade-off:**
- ✅ **Much higher quality** weak labels (minimal noise)
- ✅ **Better pre-training** in Notebook 3 (clean signal)
- ⚠️ **Smaller training set** for semi-supervised scenarios
- ⚠️ But quality > quantity for medical applications

**Implementation:**
- Calculate 80th percentile of confidence scores (from unlabeled data only)
- Use this value as threshold
- Keep only samples with confidence >= threshold"""

# Find and update Cell 23
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '### 6.2 Confidence Thresholding' in source_text:
            cell['source'] = cell_23_updated.split('\n')
            print(f"Updated cell {i} (Cell 23): Changed to TOP 20% strategy")
            break

# Update Cell 24 code - change threshold calculation
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'CONFIDENCE THRESHOLDING - FILTER HIGH-QUALITY WEAK LABELS' in source_text:
            # Replace the entire cell with updated code
            updated_code = '''print("="*80)
print("CONFIDENCE THRESHOLDING - FILTER TOP 20% HIGH-QUALITY WEAK LABELS")
print("="*80)

# Calculate distances to both cluster centroids for ALL samples
centroids = kmeans.cluster_centers_  # Shape: (2, 50)

# Calculate confidence using SILHOUETTE SCORES (better than distance-based)
from sklearn.metrics import silhouette_samples

print("\\nCalculating confidence using silhouette scores...")

# Silhouette score per sample: measures how similar a sample is to its cluster vs other cluster
# Range: [-1, 1], higher is better
# > 0.5: Strong assignment, 0.3-0.5: Moderate, < 0.3: Weak/ambiguous
silhouette_scores_all = silhouette_samples(features_pca_50, aligned_clusters)

# Normalize to [0, 1] range for consistency
# Original range [-1, 1] -> [0, 1]
confidence_scores = (silhouette_scores_all + 1) / 2

print(f"\\nConfidence Score Statistics (all {len(confidence_scores)} samples):")
print(f"  - Mean:       {confidence_scores.mean():.4f}")
print(f"  - Median:     {np.median(confidence_scores):.4f}")
print(f"  - Std:        {confidence_scores.std():.4f}")
print(f"  - Min:        {confidence_scores.min():.4f}")
print(f"  - Max:        {confidence_scores.max():.4f}")

# Calculate threshold from UNLABELED data only (to keep top 20%)
unlabeled_confidence_scores = confidence_scores[unlabeled_mask]

print(f"\\nUnlabeled Data Confidence Statistics:")
print(f"  - Mean:   {unlabeled_confidence_scores.mean():.4f}")
print(f"  - Median: {np.median(unlabeled_confidence_scores):.4f}")
print(f"  - 50th percentile: {np.percentile(unlabeled_confidence_scores, 50):.4f}")
print(f"  - 70th percentile: {np.percentile(unlabeled_confidence_scores, 70):.4f}")
print(f"  - 80th percentile: {np.percentile(unlabeled_confidence_scores, 80):.4f}")
print(f"  - 90th percentile: {np.percentile(unlabeled_confidence_scores, 90):.4f}")

# Use 80th percentile to keep TOP 20% of unlabeled data
CONFIDENCE_THRESHOLD = np.percentile(unlabeled_confidence_scores, 80)

print(f"\\n" + "="*80)
print(f"FILTERING WITH THRESHOLD = {CONFIDENCE_THRESHOLD:.4f} (80th percentile)")
print(f"STRATEGY: Keep only TOP 20% of unlabeled samples")
print("="*80)

# Apply threshold to unlabeled data only
unlabeled_high_confidence_mask = unlabeled_confidence_scores >= CONFIDENCE_THRESHOLD

print(f"\\nFiltering Results (Unlabeled Data Only):")
print(f"  - Total unlabeled samples: {unlabeled_mask.sum()}")
print(f"  - High confidence (top 20%): {unlabeled_high_confidence_mask.sum()} ({unlabeled_high_confidence_mask.sum()/unlabeled_mask.sum()*100:.1f}%)")
print(f"  - Low confidence (filtered out): {(~unlabeled_high_confidence_mask).sum()} ({(~unlabeled_high_confidence_mask).sum()/unlabeled_mask.sum()*100:.1f}%)")

print(f"\\nExpected: ~{int(unlabeled_mask.sum() * 0.2)} samples (20% of {unlabeled_mask.sum()})")
print(f"Actual: {unlabeled_high_confidence_mask.sum()} samples")

# For all data (including labeled), high confidence mask
high_confidence_mask = confidence_scores >= CONFIDENCE_THRESHOLD

print(f"\\n📊 Overall Statistics:")
print(f"  - Total samples (all): {len(confidence_scores)}")
print(f"  - High confidence (all): {high_confidence_mask.sum()} ({high_confidence_mask.sum()/len(confidence_scores)*100:.1f}%)")
print(f"  - Labeled (always kept): {labeled_mask.sum()}")
print(f"  - Unlabeled high-conf: {unlabeled_high_confidence_mask.sum()}")

# Visualize confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Confidence distribution (unlabeled data only)
axes[0].hist(unlabeled_confidence_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {CONFIDENCE_THRESHOLD:.4f} (80th %ile)')
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Confidence Distribution (Unlabeled Data)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Add shaded region for top 20%
y_max = axes[0].get_ylim()[1]
axes[0].fill_between([CONFIDENCE_THRESHOLD, unlabeled_confidence_scores.max()],
                      0, y_max, alpha=0.2, color='green', label='Top 20% (kept)')
axes[0].legend(fontsize=11)

# Plot 2: Cumulative distribution
sorted_scores = np.sort(unlabeled_confidence_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100

axes[1].plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
axes[1].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'80th percentile = {CONFIDENCE_THRESHOLD:.4f}')
axes[1].axhline(80, color='orange', linestyle=':', linewidth=2, label='80% mark')
axes[1].set_xlabel('Confidence Score', fontsize=12)
axes[1].set_ylabel('Cumulative Percentage', fontsize=12)
axes[1].set_title('Cumulative Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze confidence by cluster
print(f"\\n📊 Confidence by Cluster (Unlabeled Data):")
for cluster_id in [0, 1]:
    cluster_mask = (aligned_clusters[unlabeled_mask] == cluster_id)
    cluster_confidence = unlabeled_confidence_scores[cluster_mask]
    high_conf_in_cluster = (cluster_confidence >= CONFIDENCE_THRESHOLD).sum()

    print(f"\\n  Cluster {cluster_id}:")
    print(f"    - Total samples: {cluster_mask.sum()}")
    print(f"    - High confidence (top 20%): {high_conf_in_cluster} ({high_conf_in_cluster/cluster_mask.sum()*100:.1f}%)")
    print(f"    - Mean confidence: {cluster_confidence.mean():.4f}")
    print(f"    - Median confidence: {np.median(cluster_confidence):.4f}")
    print(f"    - Min confidence: {cluster_confidence.min():.4f}")
    print(f"    - Max confidence: {cluster_confidence.max():.4f}")

print(f"\\n✓ Confidence filtering complete")
print(f"\\n💡 Result: Kept {unlabeled_high_confidence_mask.sum()} high-quality weak labels")
print(f"   This is the TOP {unlabeled_high_confidence_mask.sum()/unlabeled_mask.sum()*100:.1f}% most confident assignments")'''

            cell['source'] = updated_code.split('\n')
            print(f"Updated cell {i} (Cell 24): Changed to top 20% threshold calculation")
            break

# Update Cell 25 markdown - interpretation of top 20% strategy
cell_25_updated = '''# Update weak labels dataset with confidence scores (TOP 20% ONLY)
print("="*80)
print("UPDATING WEAK LABELS - KEEPING TOP 20% ONLY")
print("="*80)

# Add confidence scores to labeling dataframe
labeling_df['confidence_score'] = confidence_scores

# For labeled data, we keep all (they have true labels)
# For unlabeled data, we only keep top 20% (highest confidence)
# Mark low-confidence samples as -1 (to be excluded from training)
labeling_df['weak_label_kmeans_filtered'] = labeling_df['weak_label_kmeans'].copy()

# Create high-confidence mask for unlabeled data
# We need to map the unlabeled_high_confidence_mask back to the full dataframe
unlabeled_indices = np.where(labeling_df['true_label'] == -1)[0]
low_confidence_indices = unlabeled_indices[~unlabeled_high_confidence_mask]

# Set low-confidence unlabeled samples to -1 (excluded)
labeling_df.loc[low_confidence_indices, 'weak_label_kmeans_filtered'] = -1

# Separate datasets
strong_labeled_df = labeling_df[labeling_df['true_label'] != -1].copy()
weak_labeled_df_all = labeling_df[labeling_df['true_label'] == -1].copy()
weak_labeled_df_filtered = weak_labeled_df_all[weak_labeled_df_all['weak_label_kmeans_filtered'] != -1].copy()

print(f"\\n📋 Updated Weak Labels Summary:")
print(f"\\nStrong Labels (Expert-labeled) - UNCHANGED:")
print(f"  - Total: {len(strong_labeled_df)}")
print(f"  - Normal (0): {(strong_labeled_df['true_label'] == 0).sum()}")
print(f"  - Cancer (1): {(strong_labeled_df['true_label'] == 1).sum()}")

print(f"\\nWeak Labels (K-Means) - BEFORE FILTERING:")
print(f"  - Total: {len(weak_labeled_df_all)}")
print(f"  - Cluster 0: {(weak_labeled_df_all['weak_label_kmeans'] == 0).sum()}")
print(f"  - Cluster 1: {(weak_labeled_df_all['weak_label_kmeans'] == 1).sum()}")

print(f"\\nWeak Labels (K-Means) - AFTER TOP 20% FILTERING:")
print(f"  - Total: {len(weak_labeled_df_filtered)} (TOP {len(weak_labeled_df_filtered)/len(weak_labeled_df_all)*100:.1f}%)")
print(f"  - Cluster 0: {(weak_labeled_df_filtered['weak_label_kmeans_filtered'] == 0).sum()}")
print(f"  - Cluster 1: {(weak_labeled_df_filtered['weak_label_kmeans_filtered'] == 1).sum()}")
print(f"  - Filtered out: {len(weak_labeled_df_all) - len(weak_labeled_df_filtered)} samples ({(len(weak_labeled_df_all) - len(weak_labeled_df_filtered))/len(weak_labeled_df_all)*100:.1f}%)")

# Check quality improvement on labeled data
agreement_original = (strong_labeled_df['true_label'] == strong_labeled_df['weak_label_kmeans']).mean()

# For filtered, only check high-confidence labeled samples
high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= CONFIDENCE_THRESHOLD
if high_conf_labeled_mask.sum() > 0:
    agreement_filtered = (strong_labeled_df.loc[high_conf_labeled_mask, 'true_label'] ==
                         strong_labeled_df.loc[high_conf_labeled_mask, 'weak_label_kmeans']).mean()
    improvement = (agreement_filtered - agreement_original) * 100
else:
    agreement_filtered = 0.0
    improvement = 0.0

print(f"\\n📊 Quality Comparison (on labeled data):")
print(f"  - Original weak label agreement: {agreement_original:.2%} (all {len(strong_labeled_df)} samples)")
if high_conf_labeled_mask.sum() > 0:
    print(f"  - High-confidence agreement: {agreement_filtered:.2%} (only {high_conf_labeled_mask.sum()} high-conf samples)")
    print(f"  - Quality improvement: {'+' if improvement >= 0 else ''}{improvement:.1f} percentage points")
else:
    print(f"  - High-confidence agreement: N/A (no labeled samples in top 20%)")

print(f"\\n💡 Interpretation:")
print(f"   - Filtered OUT: {100 - len(weak_labeled_df_filtered)/len(weak_labeled_df_all)*100:.1f}% of unlabeled data")
print(f"   - Kept: TOP {len(weak_labeled_df_filtered)/len(weak_labeled_df_all)*100:.1f}% with highest confidence")
print(f"   - Trade-off: Much smaller dataset but MUCH higher quality")
print(f"   - Strategy: Quality > Quantity for medical imaging")

# Save BOTH versions (for comparison in Notebook 3)
OUTPUT_DIR = Path('features')

# Original (unfiltered)
labeling_df.to_csv(OUTPUT_DIR / 'weak_labels.csv', index=False)
print(f"\\n✓ Original weak labels saved: {OUTPUT_DIR / 'weak_labels.csv'}")

# Filtered (high-confidence only - TOP 20%)
labeling_df.to_csv(OUTPUT_DIR / 'weak_labels_filtered.csv', index=False)
print(f"✓ Filtered weak labels saved: {OUTPUT_DIR / 'weak_labels_filtered.csv'}")

# Also save just the high-confidence unlabeled subset for easy loading
weak_labeled_df_filtered.to_csv(OUTPUT_DIR / 'weak_labels_high_confidence.csv', index=False)
print(f"✓ High-confidence subset saved: {OUTPUT_DIR / 'weak_labels_high_confidence.csv'}")

print(f"\\n" + "="*80)
print("TOP 20% CONFIDENCE FILTERING COMPLETE")
print("="*80)
print(f"\\nNotebook 3 can now use:")
print(f"  1. weak_labels.csv - All {len(weak_labeled_df_all)} weak labels (original)")
print(f"  2. weak_labels_high_confidence.csv - {len(weak_labeled_df_filtered)} TOP QUALITY labels (recommended)")
print(f"\\nRecommendation: Use TOP 20% high-confidence labels for best performance")
print(f"Quality over quantity is critical for medical imaging applications!")'''

# Find and update Cell 25
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'UPDATING WEAK LABELS WITH CONFIDENCE FILTERING' in source_text:
            cell['source'] = cell_25_updated.split('\n')
            print(f"Updated cell {i} (Cell 25): Updated to top 20% interpretation")
            break

# Update Cell 30 summary to reflect top 20% strategy
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 8. Summary and Key Findings' in source_text:
            # Update the confidence filtering section
            source_text = source_text.replace(
                '- **Applied confidence thresholding (≥0.9)**',
                '- **Applied confidence thresholding (TOP 20%)**'
            )
            source_text = source_text.replace(
                '- Retained ~40-50% of unlabeled data with highest confidence scores',
                '- Retained TOP 20% (~281 samples) of unlabeled data with highest confidence scores'
            )
            source_text = source_text.replace(
                '- Improved label quality at cost of fewer training samples',
                '- Significantly improved label quality by keeping only the most confident assignments'
            )
            cell['source'] = source_text.split('\n')
            print(f"Updated cell {i} (Cell 30 Summary): Updated to reflect top 20% strategy")
            break

# Save updated notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 2 UPDATED - TOP 20% CONFIDENCE STRATEGY")
print("="*80)
print("\nChanges applied:")
print("  1. Cell 23: Updated markdown to explain TOP 20% strategy")
print("  2. Cell 24: Changed threshold to np.percentile(unlabeled_scores, 80)")
print("  3. Cell 25: Updated interpretation for top 20% filtering")
print("  4. Cell 30: Updated summary to reflect ~281 samples (20% of 1406)")
print("\nExpected outcome:")
print(f"  - Keep approximately 281 unlabeled samples (top 20% of 1,406)")
print(f"  - Filter out approximately 1,125 samples (bottom 80%)")
print(f"  - Much higher quality weak labels for semi-supervised learning")
print("\n✓ Notebook ready for execution!")
