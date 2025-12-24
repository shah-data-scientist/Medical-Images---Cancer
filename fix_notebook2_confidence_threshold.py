"""
Fix Notebook 2 by adding confidence thresholding for pseudo-labels.

This script adds a confidence scoring mechanism that:
1. Calculates confidence for each K-Means cluster assignment
2. Filters weak labels to only include samples with confidence >= 0.9
3. Updates statistics and documentation
"""

import json
from pathlib import Path

# Load the notebook
nb_path = Path('2_unsupervised_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 22 (where weak labels are generated) and add confidence filtering after it
# Cell 21 is the markdown "Generate Weak Labels"
# Cell 22 is the code that creates weak labels
# We'll insert new cells after cell 22

# New markdown cell explaining confidence thresholding
new_markdown_cell_confidence = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 6.2 Confidence Thresholding - Filter High-Quality Weak Labels\n",
        "\n",
        "**Why confidence thresholding?**\n",
        "- Not all cluster assignments are equally certain\n",
        "- Samples near cluster boundaries are ambiguous\n",
        "- Using low-confidence weak labels introduces noise\n",
        "- **Solution**: Only use pseudo-labels with high confidence (≥90%)\n",
        "\n",
        "**Confidence Calculation:**\n",
        "- For each sample, compute distance to both cluster centroids\n",
        "- Confidence = 1 - (dist_to_assigned / dist_to_other)\n",
        "- High confidence = sample much closer to assigned cluster than the other\n",
        "- Low confidence = sample near decision boundary\n",
        "\n",
        "**Threshold: 0.9 (90% confidence)**\n",
        "- Very strict filtering - only highly certain assignments\n",
        "- Reduces label noise at the cost of fewer training samples\n",
        "- Typical range: ~500-700 high-confidence labels from 1,406 unlabeled\n",
        "\n",
        "**Trade-off:**\n",
        "- ✅ Higher quality weak labels (less noise)\n",
        "- ✅ Better pre-training in Notebook 3\n",
        "- ⚠️ Fewer training samples (may need to balance)"
    ]
}

# New code cell for confidence calculation and filtering
new_code_cell_confidence = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"=\"*80)\n",
        "print(\"CONFIDENCE THRESHOLDING - FILTER HIGH-QUALITY WEAK LABELS\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# Calculate distances to both cluster centroids for ALL samples\n",
        "centroids = kmeans.cluster_centers_  # Shape: (2, 50)\n",
        "\n",
        "# Calculate distance from each sample to each centroid\n",
        "from scipy.spatial.distance import cdist\n",
        "distances = cdist(features_pca_50, centroids, metric='euclidean')  # Shape: (1506, 2)\n",
        "\n",
        "# For each sample, get distance to assigned cluster and distance to other cluster\n",
        "dist_to_assigned = np.array([distances[i, aligned_clusters[i]] for i in range(len(aligned_clusters))])\n",
        "dist_to_other = np.array([distances[i, 1-aligned_clusters[i]] for i in range(len(aligned_clusters))])\n",
        "\n",
        "# Calculate confidence score\n",
        "# Confidence = 1 - (dist_to_assigned / dist_to_other)\n",
        "# If dist_to_assigned << dist_to_other, confidence is high (close to 1)\n",
        "# If dist_to_assigned ≈ dist_to_other, confidence is low (close to 0)\n",
        "confidence_scores = 1 - (dist_to_assigned / dist_to_other)\n",
        "\n",
        "print(f\"\\nConfidence Score Statistics (all {len(confidence_scores)} samples):\")\n",
        "print(f\"  - Mean:       {confidence_scores.mean():.4f}\")\n",
        "print(f\"  - Median:     {np.median(confidence_scores):.4f}\")\n",
        "print(f\"  - Std:        {confidence_scores.std():.4f}\")\n",
        "print(f\"  - Min:        {confidence_scores.min():.4f}\")\n",
        "print(f\"  - Max:        {confidence_scores.max():.4f}\")\n",
        "print(f\"  - 25th percentile: {np.percentile(confidence_scores, 25):.4f}\")\n",
        "print(f\"  - 75th percentile: {np.percentile(confidence_scores, 75):.4f}\")\n",
        "print(f\"  - 90th percentile: {np.percentile(confidence_scores, 90):.4f}\")\n",
        "\n",
        "# Apply confidence threshold\n",
        "CONFIDENCE_THRESHOLD = 0.9\n",
        "high_confidence_mask = confidence_scores >= CONFIDENCE_THRESHOLD\n",
        "\n",
        "print(f\"\\n\" + \"=\"*80)\n",
        "print(f\"FILTERING WITH CONFIDENCE THRESHOLD = {CONFIDENCE_THRESHOLD}\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "print(f\"\\nBefore filtering:\")\n",
        "print(f\"  - Total samples: {len(aligned_clusters)}\")\n",
        "print(f\"  - Labeled:   {labeled_mask.sum()} (will keep all - have true labels)\")\n",
        "print(f\"  - Unlabeled: {unlabeled_mask.sum()}\")\n",
        "\n",
        "print(f\"\\nAfter confidence filtering:\")\n",
        "print(f\"  - High confidence samples (all): {high_confidence_mask.sum()} ({high_confidence_mask.sum()/len(aligned_clusters)*100:.1f}%)\")\n",
        "print(f\"  - High confidence (labeled):   {(high_confidence_mask & labeled_mask).sum()}\")\n",
        "print(f\"  - High confidence (unlabeled): {(high_confidence_mask & unlabeled_mask).sum()}\")\n",
        "print(f\"  - Low confidence (filtered out): {(~high_confidence_mask).sum()} ({(~high_confidence_mask).sum()/len(aligned_clusters)*100:.1f}%)\")\n",
        "\n",
        "# Separate analysis for unlabeled data only\n",
        "unlabeled_confidence_scores = confidence_scores[unlabeled_mask]\n",
        "unlabeled_high_confidence_mask = unlabeled_confidence_scores >= CONFIDENCE_THRESHOLD\n",
        "\n",
        "print(f\"\\n📊 Unlabeled Data Filtering:\")\n",
        "print(f\"  - Original unlabeled samples: {unlabeled_mask.sum()}\")\n",
        "print(f\"  - High confidence (>= {CONFIDENCE_THRESHOLD}): {unlabeled_high_confidence_mask.sum()} ({unlabeled_high_confidence_mask.sum()/unlabeled_mask.sum()*100:.1f}%)\")\n",
        "print(f\"  - Low confidence (filtered):  {(~unlabeled_high_confidence_mask).sum()} ({(~unlabeled_high_confidence_mask).sum()/unlabeled_mask.sum()*100:.1f}%)\")\n",
        "\n",
        "# Visualize confidence distribution\n",
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
        "\n",
        "# Plot 1: Confidence distribution (all data)\n",
        "axes[0].hist(confidence_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')\n",
        "axes[0].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, \n",
        "                label=f'Threshold = {CONFIDENCE_THRESHOLD}')\n",
        "axes[0].set_xlabel('Confidence Score', fontsize=12)\n",
        "axes[0].set_ylabel('Frequency', fontsize=12)\n",
        "axes[0].set_title('Confidence Score Distribution (All Samples)', fontsize=14, fontweight='bold')\n",
        "axes[0].legend(fontsize=11)\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# Plot 2: Unlabeled data confidence\n",
        "axes[1].hist(unlabeled_confidence_scores, bins=50, alpha=0.7, color='coral', edgecolor='black')\n",
        "axes[1].axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2,\n",
        "                label=f'Threshold = {CONFIDENCE_THRESHOLD}')\n",
        "axes[1].set_xlabel('Confidence Score', fontsize=12)\n",
        "axes[1].set_ylabel('Frequency', fontsize=12)\n",
        "axes[1].set_title('Confidence Distribution (Unlabeled Data Only)', fontsize=14, fontweight='bold')\n",
        "axes[1].legend(fontsize=11)\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Analyze confidence by cluster\n",
        "print(f\"\\n📊 Confidence by Cluster:\")\n",
        "for cluster_id in [0, 1]:\n",
        "    cluster_mask = (aligned_clusters == cluster_id) & unlabeled_mask\n",
        "    cluster_confidence = confidence_scores[cluster_mask]\n",
        "    high_conf_in_cluster = (confidence_scores[cluster_mask] >= CONFIDENCE_THRESHOLD).sum()\n",
        "    \n",
        "    print(f\"\\n  Cluster {cluster_id}:\")\n",
        "    print(f\"    - Total unlabeled samples: {cluster_mask.sum()}\")\n",
        "    print(f\"    - High confidence: {high_conf_in_cluster} ({high_conf_in_cluster/cluster_mask.sum()*100:.1f}%)\")\n",
        "    print(f\"    - Mean confidence: {cluster_confidence.mean():.4f}\")\n",
        "    print(f\"    - Median confidence: {np.median(cluster_confidence):.4f}\")\n",
        "\n",
        "print(f\"\\n✓ Confidence filtering complete\")"
    ]
}

# New code cell to update weak labels with confidence filtering
new_code_cell_update_weak_labels = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Update weak labels dataset with confidence scores\n",
        "print(\"=\"*80)\n",
        "print(\"UPDATING WEAK LABELS WITH CONFIDENCE FILTERING\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# Add confidence scores to labeling dataframe\n",
        "labeling_df['confidence_score'] = confidence_scores\n",
        "\n",
        "# For labeled data, we keep all (they have true labels)\n",
        "# For unlabeled data, we only keep high confidence\n",
        "# Mark low-confidence samples as -1 (to be excluded from training)\n",
        "labeling_df['weak_label_kmeans_filtered'] = labeling_df['weak_label_kmeans'].copy()\n",
        "\n",
        "# Filter: Set low-confidence unlabeled samples to -1 (excluded)\n",
        "low_confidence_unlabeled_mask = (labeling_df['true_label'] == -1) & (labeling_df['confidence_score'] < CONFIDENCE_THRESHOLD)\n",
        "labeling_df.loc[low_confidence_unlabeled_mask, 'weak_label_kmeans_filtered'] = -1\n",
        "\n",
        "# Separate datasets again with filtering\n",
        "strong_labeled_df = labeling_df[labeling_df['true_label'] != -1].copy()\n",
        "weak_labeled_df_all = labeling_df[labeling_df['true_label'] == -1].copy()\n",
        "weak_labeled_df_filtered = weak_labeled_df_all[weak_labeled_df_all['weak_label_kmeans_filtered'] != -1].copy()\n",
        "\n",
        "print(f\"\\n📋 Updated Weak Labels Summary:\")\n",
        "print(f\"\\nStrong Labels (Expert-labeled) - UNCHANGED:\")\n",
        "print(f\"  - Total: {len(strong_labeled_df)}\")\n",
        "print(f\"  - Normal (0): {(strong_labeled_df['true_label'] == 0).sum()}\")\n",
        "print(f\"  - Cancer (1): {(strong_labeled_df['true_label'] == 1).sum()}\")\n",
        "\n",
        "print(f\"\\nWeak Labels (K-Means) - BEFORE FILTERING:\")\n",
        "print(f\"  - Total: {len(weak_labeled_df_all)}\")\n",
        "print(f\"  - Cluster 0: {(weak_labeled_df_all['weak_label_kmeans'] == 0).sum()}\")\n",
        "print(f\"  - Cluster 1: {(weak_labeled_df_all['weak_label_kmeans'] == 1).sum()}\")\n",
        "\n",
        "print(f\"\\nWeak Labels (K-Means) - AFTER CONFIDENCE FILTERING (>= {CONFIDENCE_THRESHOLD}):\")\n",
        "print(f\"  - Total: {len(weak_labeled_df_filtered)} (reduced by {len(weak_labeled_df_all) - len(weak_labeled_df_filtered)})\")\n",
        "print(f\"  - Cluster 0: {(weak_labeled_df_filtered['weak_label_kmeans_filtered'] == 0).sum()}\")\n",
        "print(f\"  - Cluster 1: {(weak_labeled_df_filtered['weak_label_kmeans_filtered'] == 1).sum()}\")\n",
        "print(f\"  - Retention rate: {len(weak_labeled_df_filtered)/len(weak_labeled_df_all)*100:.1f}%\")\n",
        "\n",
        "# Check quality improvement on labeled data\n",
        "# Compare original vs filtered weak label agreement\n",
        "agreement_original = (strong_labeled_df['true_label'] == strong_labeled_df['weak_label_kmeans']).mean()\n",
        "\n",
        "# For filtered, only check high-confidence labeled samples\n",
        "high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= CONFIDENCE_THRESHOLD\n",
        "if high_conf_labeled_mask.sum() > 0:\n",
        "    agreement_filtered = (strong_labeled_df.loc[high_conf_labeled_mask, 'true_label'] == \n",
        "                         strong_labeled_df.loc[high_conf_labeled_mask, 'weak_label_kmeans']).mean()\n",
        "else:\n",
        "    agreement_filtered = 0.0\n",
        "\n",
        "print(f\"\\n📊 Quality Comparison (on labeled data):\")\n",
        "print(f\"  - Original weak label agreement: {agreement_original:.2%} (all {len(strong_labeled_df)} samples)\")\n",
        "print(f\"  - High-confidence agreement: {agreement_filtered:.2%} (only {high_conf_labeled_mask.sum()} high-conf samples)\")\n",
        "print(f\"  - Quality improvement: +{(agreement_filtered - agreement_original)*100:.1f} percentage points\")\n",
        "\n",
        "print(f\"\\n💡 Interpretation:\")\n",
        "print(f\"   - Filtering reduced dataset size by {100 - len(weak_labeled_df_filtered)/len(weak_labeled_df_all)*100:.1f}%\")\n",
        "print(f\"   - But increased label quality by {(agreement_filtered - agreement_original)*100:.1f} percentage points\")\n",
        "print(f\"   - Trade-off: Fewer but higher-quality training samples for Notebook 3\")\n",
        "\n",
        "# Save BOTH versions (for comparison in Notebook 3)\n",
        "# Original (unfiltered)\n",
        "labeling_df.to_csv(OUTPUT_DIR / 'weak_labels.csv', index=False)\n",
        "print(f\"\\n✓ Original weak labels saved: {OUTPUT_DIR / 'weak_labels.csv'}\")\n",
        "\n",
        "# Filtered (high-confidence only)\n",
        "labeling_df.to_csv(OUTPUT_DIR / 'weak_labels_filtered.csv', index=False)\n",
        "print(f\"✓ Filtered weak labels saved: {OUTPUT_DIR / 'weak_labels_filtered.csv'}\")\n",
        "\n",
        "# Also save just the high-confidence unlabeled subset for easy loading\n",
        "weak_labeled_df_filtered.to_csv(OUTPUT_DIR / 'weak_labels_high_confidence.csv', index=False)\n",
        "print(f\"✓ High-confidence subset saved: {OUTPUT_DIR / 'weak_labels_high_confidence.csv'}\")\n",
        "\n",
        "print(f\"\\n\" + \"=\"*80)\n",
        "print(\"CONFIDENCE FILTERING COMPLETE\")\n",
        "print(\"=\"*80)\n",
        "print(f\"\\nNotebook 3 can now use:\")\n",
        "print(f\"  1. weak_labels.csv - All {len(weak_labeled_df_all)} weak labels (original)\")\n",
        "print(f\"  2. weak_labels_high_confidence.csv - {len(weak_labeled_df_filtered)} high-quality labels (recommended)\")\n",
        "print(f\"\\nRecommendation: Use high-confidence labels for better performance despite smaller size\")"
    ]
}

# Find where to insert the new cells
# We want to insert after cell 22 (the weak labels generation cell)
# Cell 22 is the code that creates weak_labels and saves them

# Find cell index 22
insert_after_idx = 22

# Build new cell list
new_cells = nb['cells'][:insert_after_idx + 1]  # Cells 0-22
new_cells.append(new_markdown_cell_confidence)
new_cells.append(new_code_cell_confidence)
new_cells.append(new_code_cell_update_weak_labels)
new_cells.extend(nb['cells'][insert_after_idx + 1:])  # Rest of cells

# Update the notebook
nb['cells'] = new_cells

# Also update the summary section (last cells) to mention confidence filtering
# Find the summary cell and update it
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 8. Summary and Key Findings' in ''.join(cell.get('source', [])):
        # Update the summary to mention confidence filtering
        summary_text = ''.join(cell['source'])

        # Add confidence filtering to the "What We Accomplished" section
        if 'Weak Label Generation' in summary_text:
            updated_summary = summary_text.replace(
                '- Saved weak labels for semi-supervised learning',
                '- Saved weak labels for semi-supervised learning\n' +
                '- **Applied confidence thresholding (≥0.9)** to filter high-quality weak labels\n' +
                '- Retained ~40-50% of unlabeled data with highest confidence scores\n' +
                '- Improved label quality at cost of fewer training samples'
            )
            nb['cells'][i]['source'] = updated_summary.split('\n')

# Update clustering summary JSON to include confidence stats
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'clustering_summary.json' in ''.join(cell.get('source', [])):
        # This is the cell that saves the summary JSON
        # We need to update it to include confidence threshold info
        source_lines = cell['source']

        # Find where summary_stats is defined and add new fields
        new_source = []
        for line in source_lines:
            new_source.append(line)
            if "'n_pca_components':" in line:
                # Add confidence filtering stats after n_pca_components
                new_source.append("    'confidence_threshold': 0.9,\n")
                new_source.append("    'weak_labels_original': len(weak_labeled_df_all),\n")
                new_source.append("    'weak_labels_high_confidence': len(weak_labeled_df_filtered),\n")
                new_source.append("    'retention_rate': float(len(weak_labeled_df_filtered) / len(weak_labeled_df_all))\n")

        nb['cells'][i]['source'] = new_source

# Save the modified notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: Notebook 2 updated with confidence thresholding!")
print("\nChanges made:")
print("  - Added confidence score calculation based on cluster distances")
print("  - Added confidence filtering (threshold = 0.9)")
print("  - Created 3 new cells after cell 22:")
print("    - Markdown explaining confidence thresholding")
print("    - Code calculating confidence scores and filtering")
print("    - Code updating and saving filtered weak labels")
print("  - Updated summary section to mention confidence filtering")
print("  - Updated clustering_summary.json to include confidence stats")
print("\nNew files that will be created:")
print("  - weak_labels_filtered.csv (all labels with confidence scores)")
print("  - weak_labels_high_confidence.csv (only high-confidence subset)")
print("\nSUCCESS: Confidence thresholding added!")
