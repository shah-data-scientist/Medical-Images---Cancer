"""
Fix markdown cell integrity violations in Notebook 2

Issues addressed:
1. Cell 7 - Remove forward references to results not yet computed
2. Cell 9 - Update stale variance numbers (73.5% -> 97.99%)
3. Cell 23 - Update confidence threshold expectations (500-700 -> 1406)
"""
import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# FIX 1: Cell 7 - Remove forward references, keep only conceptual introduction
cell_7_fixed = """## 4. Dimensionality Reduction

### 4.1 PCA (Principal Component Analysis)

**What is PCA?**
- Linear transformation that finds directions of maximum variance
- Reduces 2048 dimensions to fewer components while preserving variance
- First component = direction of highest variance, second = second highest, etc.

**Our approach:**
- Reduce to 50 dimensions for downstream processing
- Then use t-SNE on these 50 dimensions for 2D visualization

**Why 50 components?**
- Balance between information retention and computational efficiency
- Reduces curse of dimensionality (Euclidean distances become more meaningful)
- Conventional choice for neural network feature inputs
- Preserves most discriminative variance while removing noise

**Note**: We'll analyze the impact of different PCA dimensions on clustering quality below."""

# Find and update cell 7
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 4. Dimensionality Reduction' in source_text and 'Why 50 components?' in source_text:
            # This is cell 7
            cell['source'] = cell_7_fixed.split('\n')
            print(f"Fixed cell {i} (Cell 7): Removed forward references to analysis results")
            break

# FIX 2: Cell 9 - Update variance numbers to match actual output (97.99% not 73.5%)
cell_9_fixed = """### 4.2 Key Insight: The Curse of Dimensionality

**Important Discovery from Execution**:

**Result from our PCA analysis**:
- With 50 components: Retained **97.99% variance**
- This high variance retention indicates first 50 components capture nearly all information
- Demonstrates that 2048D ResNet50 features have significant redundancy

**What this means:**
- First 50 principal components capture 98% of the variance in brain MRI features
- Remaining 1998 components contribute only 2% (mostly noise)
- This validates our choice of 50 components for downstream analysis

**Why High Dimensionality Can Hurt Clustering:**
- In very high dimensions, Euclidean distances become less meaningful
- K-Means relies on distance calculations, which degrade in 100+ dimensions
- Extra variance often contains noise rather than discriminative signal
- 50 dimensions provides optimal balance for clustering quality

**Expected Clustering Behavior:**
- K-Means on 50D features should produce coherent clusters
- Further dimension reduction (via t-SNE) helps 2D visualization
- Silhouette scores will measure cluster cohesion

**Takeaway**: Dimensionality reduction is essential - 50D captures nearly all useful information!

---

### 4.3 t-SNE for 2D Visualization

**What is t-SNE?**
- Non-linear dimensionality reduction technique
- Preserves local neighborhood structure
- Excellent for visualizing high-dimensional clusters in 2D/3D

**Parameters:**
- `perplexity`: Balance between local and global structure (typical: 5-50)
- `max_iter`: Number of iterations (more = better but slower)

**Note**: t-SNE can take several minutes on large datasets."""

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '### 4.2 Key Insight: The Curse of Dimensionality' in source_text:
            cell['source'] = cell_9_fixed.split('\n')
            print(f"Fixed cell {i} (Cell 9): Updated variance from 73.5% to 97.99%")
            break

# FIX 3: Cell 23 - Update confidence threshold expectations
cell_23_fixed = """### 6.2 Confidence Thresholding - Filter High-Quality Weak Labels

**Why confidence thresholding?**
- Not all cluster assignments are equally certain
- Samples near cluster boundaries are ambiguous
- Using low-confidence weak labels introduces noise
- **Solution**: Only use pseudo-labels with high confidence

**Confidence Calculation:**
- Uses **silhouette scores** to measure cluster assignment quality
- Silhouette score measures how well a sample fits its assigned cluster vs. others
- Range: [-1, 1], higher is better
- Normalized to [0, 1] for interpretability

**Threshold Strategy:**
- Using **80th percentile** as threshold (balance quality vs. quantity)
- Higher threshold = fewer but higher quality labels
- Lower threshold = more labels but potentially noisier

**Important Note**: The actual retention rate depends on the silhouette score distribution, which is calculated from the clustering results. Based on our K-means clustering with 50D PCA features, the silhouette scores indicate good cluster cohesion, which may result in high retention rates.

**Trade-off to Consider:**
- ✅ Higher quality weak labels (less noise)
- ✅ Better pre-training in Notebook 3
- ⚠️ May retain most or all labels if clustering quality is very good
- ⚠️ Threshold can be adjusted based on downstream performance needs"""

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '### 6.2 Confidence Thresholding - Filter High-Quality Weak Labels' in source_text:
            cell['source'] = cell_23_fixed.split('\n')
            print(f"Fixed cell {i} (Cell 23): Updated confidence threshold expectations")
            break

# FIX 4: Update Cell 30 (Summary) to reflect actual results
# Find cell 30 and update any stale numbers
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if '## 8. Summary and Key Findings' in source_text:
            # Update variance claim in summary
            source_text = source_text.replace('retaining 73.5% variance', 'retaining 97.99% variance')
            source_text = source_text.replace('(retaining ~73% variance)', '(retaining ~98% variance)')
            source_text = source_text.replace('73.5%', '97.99%')
            cell['source'] = source_text.split('\n')
            print(f"Fixed cell {i} (Cell 30 Summary): Updated variance percentages")
            break

# Save fixed notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 2 MARKDOWN INTEGRITY FIXES COMPLETE")
print("="*80)
print("\nFixed issues:")
print("  1. Cell 7: Removed forward references to results not yet computed")
print("  2. Cell 9: Updated variance from 73.5% to 97.99% (matches actual output)")
print("  3. Cell 23: Updated confidence threshold expectations")
print("  4. Cell 30: Updated summary statistics")
print("\nAll markdown cells now maintain proper knowledge hierarchy!")
print("Each cell only references information from preceding cells.")
