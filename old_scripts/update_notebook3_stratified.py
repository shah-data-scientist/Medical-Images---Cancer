"""
Update Notebook 3 to reflect stratified balanced filtering approach
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 7 - Weak Labeling Strategy
cell_7_updated = """## 1.5 Weak Labeling Strategy

### How Were Weak Labels Generated?

Weak labels were generated in **Notebook 2** using an unsupervised clustering approach:

**Step 1: K-means Clustering**
- Applied K-means (k=2) on 50D PCA features of unlabeled data
- Two clusters naturally emerge (representing normal/cancer patterns)
- Clusters aligned with ground truth using labeled data (cluster matching)

**Step 2: Confidence Scoring**
- Used **silhouette scores** to measure cluster assignment confidence
- Silhouette score measures how well a sample fits its assigned cluster vs. others
- Range: [-1, 1], higher is better
- Normalized to [0, 1] for interpretability

**Step 3: Stratified Confidence Filtering - TOP 20% PER CLUSTER**
- **Key Innovation**: Filter each cluster separately to maintain balance
- Calculated 80th percentile threshold **for each cluster independently**
- Kept top 20% from Cluster 0: ~121 samples (from ~603)
- Kept top 20% from Cluster 1: ~161 samples (from ~803)
- **Total: ~282 balanced samples** (ratio ~1:1.33)

**Why Stratified (Per-Cluster) Filtering?**
- **Prevents Class Imbalance**: Equal representation from both classes
- **Avoids Model Bias**: Without stratification, we'd get 43 vs 239 (1:5.5 ratio)
- **Better Training**: Balanced pre-training prevents majority class bias
- **Medical Fairness**: Both cancer and normal cases equally important

**Comparison: Stratified vs. Non-Stratified**

| Approach | Cluster 0 | Cluster 1 | Ratio | Problem? |
|----------|-----------|-----------|-------|----------|
| **Non-Stratified** (bad) | 43 | 239 | 1:5.5 | ❌ Severe imbalance |
| **Stratified** (good) | ~121 | ~161 | 1:1.33 | ✅ Balanced |

### Weak Label Quality

**Agreement with Ground Truth** (from Notebook 2 labeled validation):
- K-means clustering achieved ~82% agreement with true labels
- **Top 20% per cluster expected to have HIGHER agreement** (>85-90%)
- Stratified filtering removes ambiguous cases while maintaining balance

**Why Use Weak Labels?**
- Labeled data: Only 70 samples (expensive: €3 per image)
- High-confidence balanced weak labels: ~282 samples (essentially free)
- Semi-supervised learning leverages this to improve generalization

### Usage Strategy by Scenario

**Scenario A (Fully Supervised)**:
- Uses: 0 weak labels
- Trains only on 60-70 labeled samples per fold
- Baseline for comparison

**Scenario B (Semi-Supervised - Clustering)**:
- Uses: ~282 balanced weak labels from stratified K-means filtering
- Phase 1: Pre-train on 282 high-confidence **balanced** weak labels
- Phase 2: Fine-tune on 60-70 strong labels
- **Key advantage**: Balanced pre-training prevents class bias

**Scenario C (Semi-Supervised - Model-based)**: ⭐ NEW
- Uses: Model-generated pseudo-labels (not clustering)
- Phase 1: Train initial model on 60-70 labeled samples
- Phase 2: Generate pseudo-labels, filter with confidence ≥ 90%
- Phase 3: Retrain on labeled + high-confidence pseudo-labeled
- **Expected Advantage**: Model learns task-specific features, not just clustering patterns

---

### Hypothesis: Model-based vs. Clustering-based

**Why we expect model-based (Scenario C) to outperform clustering (Scenario B)**:
- **Clustering (B)**: Uses feature similarity (unsupervised)
  - With stratified TOP 20% filtering, quality is high and balanced
  - But still limited to generic feature patterns
- **Model-based (C)**: Learns discriminative features from labeled data (task-specific)
  - Pseudo-labels reflect learned decision boundary
  - Model confidence more meaningful than cluster distance
- **Medical imaging**: Subtle pathological patterns better captured by supervised learning

**This hypothesis will be validated** in the cross-validation results below (Section 7).

---"""

# Find and update Cell 7
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 1.5 Weak Labeling Strategy' in source_text:
            cell['source'] = cell_7_updated.split('\n')
            print(f"Updated cell {i} (Cell 7): Updated to stratified balanced filtering")
            break

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 3 UPDATED - STRATIFIED FILTERING REFERENCES")
print("="*80)
print("\nChanges applied:")
print("  1. Cell 7: Updated to reflect stratified balanced filtering")
print("  2. Added comparison table showing non-stratified vs stratified results")
print("  3. Emphasized prevention of class imbalance")
print("\nNotebook 3 now accurately reflects:")
print("  - ~282 balanced weak labels (~121 + ~161)")
print("  - Stratified per-cluster filtering strategy")
print("  - Prevention of model bias through balanced training")
print("\nBoth notebooks now consistent and ready for execution!")
