# System Overview

**Source:** Derived from code
**Last Verified:** 2025-12-28

---

## What This System Does

**BrainScanAI** is a brain tumor detection system that uses semi-supervised learning to classify MRI brain scans as Normal or Cancer.

### Core Functionality (Observed in Code)

The system implements a **three-stage pipeline**:

#### Stage 1: Feature Extraction
- **Entry Point:** `1_feature_extraction.ipynb`
- **Inputs:** Raw MRI images (JPEG) from `data/labelled/` and `data/unlabelled/`
- **Process:**
  - Loads pre-trained ResNet50 model (from PyTorch)
  - Extracts 2048-dimensional feature vectors from images
  - Applies PCA dimensionality reduction (2048D → 50D)
  - Preserves 97%+ variance
- **Outputs:**
  - `features/resnet50_features.npy` - Raw 2048D features
  - `features/features_pca_50.npy` - PCA-reduced 50D features
  - `features/labels.npy` - Ground truth labels
  - `features/metadata.csv` - Image metadata

#### Stage 2: Unsupervised Analysis
- **Entry Point:** `2_unsupervised_analysis.ipynb`
- **Inputs:** PCA features from Stage 1
- **Process:**
  - K-means clustering (k=2) on unlabeled data
  - Generates weak labels based on cluster assignments
  - Filters weak labels by confidence threshold
  - Creates t-SNE visualizations for cluster separation
- **Outputs:**
  - `features/weak_labels.csv` - Initial weak labels
  - `features/weak_labels_filtered.csv` - Confidence-filtered labels
  - `features/weak_labels_high_confidence.csv` - High-confidence subset
  - `features/clustering_summary.json` - Cluster statistics

#### Stage 3: Semi-Supervised Learning
- **Entry Point:** `3_semi_supervised_learning.ipynb`
- **Inputs:**
  - 100 manually labeled samples (50 Normal, 50 Cancer)
  - 2,724 unlabeled samples with weak labels
- **Process:**
  - Tests 3 scenarios:
    - **Scenario A:** Fully supervised (100 labeled only)
    - **Scenario B:** Clustering-based weak supervision
    - **Scenario C:** Model-based pseudo-labeling
  - 5-fold stratified cross-validation
  - 80/20 train/test split (80 training pool, 20 final test)
  - Model: Regularized MLP (50 input, 64 hidden, 70% dropout)
  - Training: 50 epochs, BCE loss, AdamW optimizer
- **Outputs:**
  - `detailed_cv_results.json` - Cross-validation results
  - `scenario_comparison.csv` - Performance summary
  - Trained models (not saved to disk, in-memory only)

### Validation Analysis (Standalone)

**Entry Points:**
- `advanced_validation_analysis.py` - Reusable validation functions
- `run_validation_analysis.py` - Standalone validation execution

**Analyses Performed:**
1. **Feature Importance:** Permutation-based importance for 50 PCA components
2. **t-SNE Visualization:** 2D projection of class separability
3. **Noise Robustness:** Performance under Gaussian noise (0-30%)

**Outputs:**
- `feature_importance_analysis.png`
- `tsne_visualization.png`
- `noise_robustness_test.png`
- `validation_analysis_results.json`

---

## System Architecture (Code-Derived)

```
┌─────────────────────────────────────────────────────────┐
│             INPUT: MRI Images (2,824 total)             │
│  - Labelled: 100 images (50 Normal, 50 Cancer)         │
│  - Unlabelled: 2,724 images                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Feature Extraction (1_feature_extraction)     │
│  - ResNet50 (pre-trained ImageNet)                      │
│  - PCA: 2048D → 50D (97% variance retained)            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Unsupervised Analysis (2_unsupervised)        │
│  - K-means clustering (k=2)                             │
│  - Weak label generation                                │
│  - Confidence filtering                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Semi-Supervised Learning (3_semi_supervised)  │
│  - 3 Scenarios (supervised, clustering, pseudo-labels)  │
│  - 5-fold cross-validation                              │
│  - RegularizedMLP model                                 │
│  - Performance: 96.43% ± 3.60% F2 score (Scenario A)   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  VALIDATION: Model Robustness Analysis                  │
│  - Feature importance (9 critical components)           │
│  - Class separability (t-SNE distance: 7.29)           │
│  - Noise robustness (100% retention @ 30% noise)       │
└─────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Classification Model + Validation Results      │
│  - Model: RegularizedMLP (96.43% F2)                   │
│  - Metrics: F2, Precision, Recall, Accuracy            │
│  - Visualizations: 3 PNG plots                         │
│  - Results: 3 JSON files                               │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow (Observed in Code)

### Input Data Structure
```
data/
├── labelled/
│   ├── no/      (Normal brain scans)
│   └── yes/     (Cancer brain scans)
└── unlabelled/
    ├── no/      (Unlabeled normal)
    └── yes/     (Unlabeled cancer - labels not used in training)
```

### Intermediate Features
```
features/
├── resnet50_features.npy        (2048D features, all images)
├── features_pca_50.npy          (50D PCA features)
├── labels.npy                   (Ground truth labels)
├── metadata.csv                 (Image paths, splits)
├── weak_labels.csv              (Clustering-based labels)
├── weak_labels_filtered.csv     (Confidence-filtered)
├── weak_labels_high_confidence.csv
└── clustering_summary.json
```

### Final Outputs
```
Root directory:
├── detailed_cv_results.json     (Cross-validation metrics)
├── scenario_comparison.csv      (Summary statistics)
├── validation_analysis_results.json
├── feature_importance_analysis.png
├── tsne_visualization.png
└── noise_robustness_test.png
```

---

## Runtime Behavior

### Model Training (Observed)
- **Hardware:** CPU (PyTorch 2.9.1+cpu)
- **Training Time:** ~47 minutes for semi-supervised (Scenario B/C)
- **Batch Size:** 16 (with drop_last=True to avoid BatchNorm errors)
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.05)
- **Loss:** Binary Cross-Entropy
- **Regularization:**
  - 70% dropout
  - Weight decay (L2 regularization)
  - Label smoothing (0.1)
  - Gradient clipping (max_norm=1.0)

### Performance Metrics (Code-Verified)
```
Scenario A (Fully Supervised):
- F2 Score: 96.43% ± 3.60%
- Recall: 98.00%
- Precision: 92.67%

Scenario B (Clustering-based):
- F2 Score: 89.65% ± 0.79%
- Most stable (lowest variance)

Scenario C (Pseudo-labels):
- F2 Score: 93.22% ± 4.40%
- Statistically equivalent to Scenario A (p = 0.178)
```

---

## Known Limitations (Code-Derived)

1. **No External Validation:**
   - Code shows only internal validation (80/20 split, 5-fold CV)
   - No external dataset loaded or tested

2. **CPU-Only Inference:**
   - PyTorch configured for CPU (`2.9.1+cpu`)
   - No GPU acceleration observed in code

3. **Manual Label Requirement:**
   - Code requires 100 manually labeled samples as bootstrap
   - No fully unsupervised mode

4. **Model Not Persisted:**
   - No `torch.save()` calls observed in notebooks
   - Models exist only during runtime
   - `.pth` files deleted and gitignored

5. **Fixed Architecture:**
   - Model architecture hardcoded (50 input, 64 hidden)
   - No hyperparameter tuning observed
   - No architecture search

---

## Success Criteria (Verified in Code)

The code defines success as:
1. ✅ F2 Score > 90% (achieved: 96.43%)
2. ✅ Recall > 90% (achieved: 98.00%)
3. ✅ Noise robustness: F2 > 80% @ 10% noise (achieved: 98.04%)
4. ✅ Feature importance: Identify critical components (achieved: 9 components)
5. ✅ Statistical validation: p-value testing across scenarios (achieved)

All success criteria are met in current code.

---

**Next:** See `02_HOW_TO_RUN.md` for execution instructions.
