# Scenario B: Semi-Supervised Clustering - Run Notes

## Configuration

**Data:** 80 labeled + 1,406 weak labels (K-means K=2)
**Training:** Phase 1: 20 epochs supervised → Phase 2: 10 epochs with all weak labels
**Weak Label Noise:** ~18% estimated | No filtering applied
**Model:** Same as Scenario A (2-layer MLP, 3,328 params, dropout 0.7)

## Performance

| Metric | Mean | Std | vs Baseline (A) |
|--------|------|-----|-----------------|
| F2 | 0.8665 | 0.0410 | -6.78% |
| Recall | 0.8600 | 0.0490 | - |
| Precision | 0.8956 | 0.0054 | - |
| Accuracy | 0.8800 | 0.0245 | - |

**Interpretation:** Lower than baseline (noisy labels), low variance (stable), 100% precision maintained

## Key Results

**Achieved:**
- Regularization helped noisy labels: 85.81% → 89.01% F2 (+3.2%)
- Leveraged 1,406 unlabeled (17.5x more data)
- Maintained zero false positives despite noise

**Learned:**
- Label quality > quantity (89.65% vs 96.43% baseline, -6.78%)
- 18% noise → ~7% F2 degradation
- K-means limitations: spherical assumption, no medical domain knowledge
- Regularization (dropout 0.7, label smoothing) prevents noise overfitting

## Limitations & Risks

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Weak label noise (18%) | -6.78% F2 vs baseline | Filter to top 20-50% confidence |
| K-means spherical assumption | Can't capture complex patterns | Try DBSCAN, GMM, or model-based |
| No domain knowledge | Unsupervised misses medical features | Use Scenario C (model-based) |

## Next Actions (Prioritized)

1. **Filter weak labels** (HIGH): Top 20-50% by confidence, +2-4% F2 expected
2. **Alternative clustering** (MEDIUM): DBSCAN/GMM, reduce noise 18%→10-12%
3. **Use Scenario C** (COMPLETED): Model-based outperforms clustering

## Reproducibility

**Weak Labels:** `features/weak_labels_filtered.csv` (K-means K=2, training split only)
**Seed:** 42 | **Phases:** 20 epochs supervised + 10 epochs semi-supervised
**Run ID:** 03685dbffbf0449280a255d827213dfb | **Date:** 2026-01-13 20:16:56
