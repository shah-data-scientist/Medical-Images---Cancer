"""
Add missing labeling strategy explanation to Notebook 3
"""
import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

# Find where to insert (after data loading section)
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 1. Load Data' in ''.join(cell['source']):
        # Insert after the data loading code cell
        insert_idx = i + 2  # After markdown and code cell
        break

if insert_idx:
    # Add labeling strategy explanation
    labeling_strategy_markdown = """## 1.5 Weak Labeling Strategy

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

**Step 3: Confidence Filtering**
- Applied 80th percentile threshold to keep only high-confidence predictions
- Original approach: Distance-based confidence (max 42%)
- **Improved approach**: Silhouette-based (better cluster quality measure)
- Result: ~1,400 weak labels available for semi-supervised learning

### Weak Label Quality

**Agreement with Ground Truth** (from labeled validation):
- K-means clustering achieved ~82% agreement with true labels
- High-confidence subset (after filtering) has higher accuracy

**Why Use Weak Labels?**
- Labeled data: Only 70 samples (expensive: €3 per image)
- Unlabeled data: 1,406 samples (already available)
- Semi-supervised learning leverages this abundance to improve generalization

### Usage Strategy by Scenario

**Scenario A (Fully Supervised)**:
- Uses: 0 weak labels
- Trains only on 60-70 labeled samples per fold

**Scenario B (Semi-Supervised - Clustering)**:
- Uses: All ~1,400 weak labels from K-means
- Phase 1: Pre-train on weak labels (learn general patterns)
- Phase 2: Fine-tune on strong labels (correct errors)

**Scenario C (Semi-Supervised - Model-based)**: ⭐ NEW
- Uses: Model-generated pseudo-labels (not clustering)
- Phase 1: Train initial model on labeled data
- Phase 2: Generate pseudo-labels with model confidence ≥ 90%
- Phase 3: Retrain on labeled + high-confidence pseudo-labeled
- **Advantage**: Model learns task-specific features, not just clustering patterns

---

### Key Insight

Model-based pseudo-labeling (Scenario C) typically outperforms clustering-based (Scenario B) because:
- Clustering uses only feature similarity (unsupervised)
- Model learns discriminative features from labeled data (supervised initialization)
- Model confidence reflects task-specific uncertainty
- Medical images have subtle patterns better captured by trained models

---"""

    # Insert the markdown cell
    nb['cells'].insert(insert_idx, {
        'cell_type': 'markdown',
        'metadata': {},
        'source': labeling_strategy_markdown.split('\n')
    })

    print(f"Added labeling strategy section at cell {insert_idx}")
    print(f"Total cells now: {len(nb['cells'])}")

    # Save
    json.dump(nb, open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8'), indent=2)
    print("\nLabeling strategy explanation added!")
else:
    print("Could not find insertion point")
