"""
Update Notebook 3 markdown cells to reflect TOP 20% confidence strategy
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 7 - Weak Labeling Strategy section
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

**Step 3: Confidence Filtering - TOP 20% STRATEGY**
- Calculated 80th percentile threshold from unlabeled data confidence scores
- **Kept only TOP 20%** of samples with highest confidence
- **Filtered out 80%** with lower confidence (bottom 1,125 samples)
- Result: ~281 high-quality weak labels (from 1,406 unlabeled)

**Filtering Strategy: Quality over Quantity**
- Original: 1,406 unlabeled samples available
- After filtering: ~281 samples retained (top 20%)
- Rationale: In medical imaging, label quality is more critical than quantity
- Better to pre-train on 281 clean labels than 1,406 noisy labels

### Weak Label Quality

**Agreement with Ground Truth** (from Notebook 2 labeled validation):
- K-means clustering achieved ~82% agreement with true labels
- **Top 20% subset expected to have HIGHER agreement** (>85-90%)
- Filtering removes ambiguous boundary cases

**Why Use Weak Labels?**
- Labeled data: Only 70 samples (expensive: €3 per image)
- High-confidence weak labels: ~281 samples (essentially free)
- Semi-supervised learning leverages this to improve generalization

### Usage Strategy by Scenario

**Scenario A (Fully Supervised)**:
- Uses: 0 weak labels
- Trains only on 60-70 labeled samples per fold
- Baseline for comparison

**Scenario B (Semi-Supervised - Clustering)**:
- Uses: ~281 top 20% weak labels from K-means
- Phase 1: Pre-train on 281 high-confidence weak labels
- Phase 2: Fine-tune on 60-70 strong labels
- Expected benefit: More data for pre-training, cleaner signal

**Scenario C (Semi-Supervised - Model-based)**: ⭐ NEW
- Uses: Model-generated pseudo-labels (not clustering)
- Phase 1: Train initial model on 60-70 labeled samples
- Phase 2: Generate pseudo-labels, filter with confidence ≥ 90%
- Phase 3: Retrain on labeled + high-confidence pseudo-labeled
- **Expected Advantage**: Model learns task-specific features, not just clustering patterns

---

### Hypothesis: Model-based vs. Clustering-based

**Why we expect model-based (Scenario C) to outperform clustering (Scenario B)**:
- **Clustering (B)**: Uses only feature similarity (unsupervised, generic patterns)
  - But with TOP 20% filtering, quality is much higher than before
- **Model-based (C)**: Learns discriminative features from labeled data first (task-specific)
  - Pseudo-labels reflect learned decision boundary
- **Medical imaging**: Subtle pathological patterns better captured by supervised learning
- **Confidence**: Model uncertainty more meaningful than cluster distance

**This hypothesis will be validated** in the cross-validation results below (Section 7).

---"""

# Find and update Cell 7
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 1.5 Weak Labeling Strategy' in source_text:
            cell['source'] = cell_7_updated.split('\n')
            print(f"Updated cell {i} (Cell 7): Changed to reflect TOP 20% filtering")
            break

# Update Cell 6 - Data loading section
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if '# Load weak labels' in source_text and 'weak_labels_high_confidence.csv' in source_text:
            # Add comment about top 20%
            lines = cell['source']
            for j, line in enumerate(lines):
                if 'weak_labels_high_confidence.csv' in line:
                    # Insert comment above this line
                    if j > 0 and '# TOP 20%' not in lines[j-1]:
                        lines.insert(j, '# TOP 20% high-confidence weak labels (filtered in Notebook 2)\n')
                        print(f"Updated cell {i} (Cell 6 Data Loading): Added TOP 20% comment")
                        break
            cell['source'] = lines
            break

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 3 UPDATED - TOP 20% REFERENCES")
print("="*80)
print("\nChanges applied:")
print("  1. Cell 7: Updated weak labeling strategy to reflect TOP 20% filtering")
print("  2. Cell 6: Added comment about TOP 20% in data loading")
print("\nNotebook 3 now accurately reflects:")
print("  - ~281 weak labels (top 20% of 1,406)")
print("  - Quality-focused filtering strategy")
print("  - Expected higher agreement with ground truth")
print("\nBoth notebooks now consistent!")
