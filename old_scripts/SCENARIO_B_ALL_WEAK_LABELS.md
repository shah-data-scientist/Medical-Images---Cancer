# Scenario B Update: Using ALL Weak Labels (No Filtering)

**Date**: 2025-12-25
**Status**: ✅ COMPLETE

---

## Summary of Changes

Scenario B has been updated to use **ALL 1,406 weak labels** from K-means clustering instead of only the top 20% filtered labels (282 samples).

---

## What Changed

### 1. Data Loading (Cell 6)

**Before**:
```python
# Load weak labels
# TOP 20% high-confidence weak labels (filtered in Notebook 2)
weak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels_high_confidence.csv')
print(f"Weak labels available: {len(weak_labels_df)}")  # 282
```

**After**:
```python
# Load weak labels
# ALL weak labels from K-means clustering (no filtering)
weak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels.csv')
print(f"Weak labels available (K-means, unfiltered): {len(weak_labels_df)}")  # 1,406
```

### 2. Weak Labeling Strategy (Cell 7)

**Updated Scenario B description**:
- Now clearly states: "Uses ALL Weak Labels"
- Mentions: 1,406 samples with ~82% quality (~250 noisy labels)
- Class distribution: Cluster 0: ~603, Cluster 1: ~803
- Experiment goal: Compare quantity vs. quality approach

### 3. Scenario B Markdown (Cell 16)

**Updated explanation**:
- Removed references to "high-confidence" and "stratified filtering"
- Now describes using all 1,406 labels with estimated 18% noise
- Added "Experiment Rationale" section explaining quantity vs. quality trade-off
- Clearly states expected trade-offs

### 4. Scenario B Code (Cell 17)

**Key code changes**:

```python
def scenario_b_clustering_semisup(train_idx, val_idx, fold):
    """Scenario B: Semi-Supervised with ALL Clustering Weak Labels (No Filtering)"""
    with mlflow.start_run(run_name=f"ScenarioB_Fold{fold}", nested=True):
        mlflow.log_params({
            "scenario": "SemiSup_Clustering_AllLabels",
            "fold": fold,
            "filtering": "None_All_Labels"  # Changed from "Top_20_Percent_Stratified"
        })

        # Use weak_label_kmeans column (all labels, no filtering)
        weak_labels = weak_labels_df['weak_label_kmeans'].values  # All 1,406

        # Log all weak label statistics
        cluster_0_count = (weak_labels == 0).sum()  # ~603
        cluster_1_count = (weak_labels == 1).sum()  # ~803

        print(f"  Using ALL {len(weak_labels)} weak labels (unfiltered K-means)")
        # ... rest of function
```

**Removed**:
- Filtering checks (no more "WARNING: Using X weak labels")
- Balance warnings (accepting 1:1.33 ratio as is)
- High-confidence filter logic

---

## Rationale for Change

### Previous Scenario B Performance (Filtered)

With **282 high-confidence stratified weak labels**:
- F2 Score: 0.5969 ± 0.0511
- Recall: 0.5733 ± 0.0533
- Precision: 0.7160 ± 0.0424

**Problem**: Even with balanced, high-confidence filtering, Scenario B significantly underperformed compared to Scenarios A and C.

### New Experiment: Quantity vs. Quality

**Hypothesis**: Perhaps the filtered approach removed too much data. Testing if:
1. **More comprehensive coverage** (1,406 vs 282) helps the model learn better representations
2. **Quantity compensates for noise** - neural networks can be robust to label noise if there's enough data
3. **Full distribution** better represents the unlabeled data space

### Comparison Setup

| Approach | Labels | Quality | Coverage | Previous F2 |
|----------|--------|---------|----------|-------------|
| **Old (Filtered)** | 282 | 85-90% | 20% | 0.5969 |
| **New (All)** | 1,406 | 82% | 100% | *To measure* |
| **Scenario C** | ~1,100 | 90%+ | ~78% | 0.9866 |

---

## Expected Outcomes

### Optimistic Case
- More data helps model learn robust features despite noise
- Better coverage of data distribution improves generalization
- F2 score improves to 0.70-0.80 range

### Realistic Case
- Noise still hurts, but not as much as before
- F2 score: 0.60-0.70 (modest improvement or similar)
- Still underperforms Scenario C significantly

### Pessimistic Case
- 250+ noisy labels overwhelm the signal
- F2 score: 0.50-0.60 (worse than filtered)
- Confirms that quality > quantity for this task

---

## Scientific Value

This experiment helps answer:

1. **Is clustering-based weak labeling viable?**
   - If ALL labels work: Yes, with proper handling of noise
   - If filtered labels work: Yes, but only with careful curation
   - If neither work: No, task-specific pseudo-labeling (Scenario C) is necessary

2. **How important is label quality vs. quantity?**
   - Direct comparison between 282 clean vs 1,406 noisy
   - Informs future labeling strategy decisions

3. **Can neural networks tolerate label noise in medical AI?**
   - Medical tasks require high precision
   - Tests robustness of two-phase semi-supervised approach

---

## Execution Instructions

### To Run Updated Scenario B

1. **Ensure Notebook 2 has been executed**:
   - `weak_labels.csv` should exist in `features/` directory
   - Contains all 1,406 cluster assignments

2. **Execute Notebook 3**:
   ```bash
   # Open 3_semi_supervised_learning.ipynb
   # Run Cell 21 (5-fold CV) or run all cells
   ```

3. **Check Output**:
   ```
   [2/3] Scenario B: Semi-Supervised (Clustering)...
     Using ALL 1406 weak labels (unfiltered K-means)
       - Cluster 0: 603 samples (42.9%)
       - Cluster 1: 803 samples (57.1%)
       - Balance ratio: 1:1.33
     Phase 1: Pre-training on 1406 weak labels (including noise)...
     Phase 1 complete: Model pre-trained on all weak labels
     Phase 2: Fine-tuning on 56 strong labels...
     Phase 2 complete: Model fine-tuned on strong labels
         Test F2: [RESULT]
   ```

4. **Compare Results**:
   - Check Cell 23 (Results Aggregation) for final F2 score
   - Compare with previous run (F2 = 0.5969)
   - Compare with Scenario C (F2 = 0.9866)

---

## MLflow Tracking

Updated MLflow parameters for Scenario B:

```python
{
    "scenario": "SemiSup_Clustering_AllLabels",  # Changed
    "filtering": "None_All_Labels",               # Changed
    "weak_labels_total": 1406,                    # Changed from 282
    "weak_labels_cluster_0": 603,                 # Changed from 121
    "weak_labels_cluster_1": 803,                 # Changed from 161
    "balance_ratio": 1.33                         # Same
}
```

You can track this in MLflow UI:
```bash
mlflow ui
# Navigate to: http://localhost:5000
# Compare "filtering" parameter: "None_All_Labels" vs "Top_20_Percent_Stratified"
```

---

## Reverting Changes (If Needed)

If results show that unfiltered labels perform worse, revert by:

1. **Change Cell 6 back**:
   ```python
   weak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels_high_confidence.csv')
   ```

2. **Restore old Scenario B code** from git history:
   ```bash
   git checkout HEAD~1 3_semi_supervised_learning.ipynb
   ```

Or use the backup script in `old_scripts/update_scenario_b_use_all_weak_labels.py` to recreate the changes.

---

## Key Insights to Watch For

After execution, look for:

1. **Pre-training Loss**: Does it converge despite noise?
2. **Fine-tuning Performance**: Can clean labels correct the noise?
3. **Test F2 Score**: Improved, same, or worse than 0.5969?
4. **Comparison with Scenario C**: Gap narrowed or still large?

---

## Files Modified

- ✅ [3_semi_supervised_learning.ipynb](3_semi_supervised_learning.ipynb)
  - Cell 6: Data loading
  - Cell 7: Weak labeling strategy
  - Cell 16: Scenario B markdown
  - Cell 17: Scenario B code

---

**Next Action**: Execute Notebook 3 to measure the impact of using all weak labels!
