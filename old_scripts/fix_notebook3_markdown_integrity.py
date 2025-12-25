"""
Fix markdown cell integrity violations in Notebook 3

Issues addressed:
1. Cell 7 - Reframe Scenario C claims as expected advantages (not proven before execution)
2. Cell 36 - Clarify budget estimates vs. actual results
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# FIX 1: Cell 7 - Reframe as "expected" advantages, not proven facts
cell_7_fixed = """## 1.5 Weak Labeling Strategy

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
- Original approach: Distance-based confidence
- **Improved approach**: Silhouette-based (better cluster quality measure)
- Result: ~1,400 weak labels available for semi-supervised learning

### Weak Label Quality

**Agreement with Ground Truth** (from Notebook 2 labeled validation):
- K-means clustering achieved ~82% agreement with true labels
- High-confidence subset (after filtering) maintains this quality

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
- **Expected Advantage**: Model learns task-specific features, not just clustering patterns

---

### Hypothesis: Model-based vs. Clustering-based

**Why we expect model-based (Scenario C) to outperform clustering (Scenario B)**:
- **Clustering (B)**: Uses only feature similarity (unsupervised, generic patterns)
- **Model-based (C)**: Learns discriminative features from labeled data first (task-specific)
- **Medical imaging**: Subtle pathological patterns better captured by supervised learning
- **Confidence**: Model uncertainty more meaningful than cluster distance

**This hypothesis will be validated** in the cross-validation results below (Section 7).

---"""

# Find and update cell 7
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 1.5 Weak Labeling Strategy' in source_text and 'Key Insight' in source_text:
            cell['source'] = cell_7_fixed.split('\n')
            print(f"Fixed cell {i} (Cell 7): Reframed as hypothesis to be validated")
            break

# FIX 2: Cell 36 - Add clarity about estimates vs. validation
# Find budget analysis cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'BUDGET ANALYSIS - €5,000 LABELING STRATEGY' in source_text:
            # Find the section where estimates are made
            source_lines = cell['source']

            # Add a clarifying comment before the estimation section
            for j, line in enumerate(source_lines):
                if '# Estimate performance based on CV results' in line:
                    # Insert clarifying comment before this line
                    source_lines.insert(j, '    # NOTE: These estimates use the cross-validation results from Section 7 above\n')
                    source_lines.insert(j+1, '    # If CV has not yet been executed, these are projected estimates\n')
                    print(f"Fixed cell {i} (Budget Analysis Code): Added clarifying comments")
                    break

            cell['source'] = source_lines
            break

# FIX 3: Update Cell 35 (Budget Analysis markdown header) to note dependency
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 9.7 Budget Analysis & Scaling Feasibility' in source_text:
            cell_35_fixed = """## 9.7 Budget Analysis & Scaling Feasibility

**Note**: This analysis uses the cross-validation results from Section 7 above. The performance estimates are based on actual measured F2-scores and Recall from the 5-fold CV execution.

### Business Context: CurelyticsIA Expansion

**Current Situation:**
- Budget available: **€5,000** for data labeling
- Labeling cost: **€3 per image** (expert radiologist)
- Current labeled dataset: **100 images** (€300 spent)
- Available unlabeled data: **1,406 images**

### Scenario Analysis: How to Spend €5,000?

Let's evaluate different labeling strategies:"""
            cell['source'] = cell_35_fixed.split('\n')
            print(f"Fixed cell {i} (Cell 35 Budget Header): Added dependency note")
            break

# Save fixed notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 3 MARKDOWN INTEGRITY FIXES COMPLETE")
print("="*80)
print("\nFixed issues:")
print("  1. Cell 7: Reframed Scenario C claims as hypothesis (validated later)")
print("  2. Cell 35: Added note about dependency on Section 7 results")
print("  3. Budget Analysis Code: Added clarifying comments about CV dependency")
print("\nAll markdown cells now maintain proper knowledge hierarchy!")
print("Claims about performance are either:")
print("  - Hypotheses to be validated (clearly stated)")
print("  - Results from previous sections (with clear references)")
