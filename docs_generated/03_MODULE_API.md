# Module & API Documentation

**Source:** Derived from code
**Last Verified:** 2025-12-28

---

## Module: advanced_validation_analysis.py

**Purpose:** Reusable validation analysis functions for model evaluation

**Imports:**
- numpy, pandas, matplotlib, seaborn
- sklearn.manifold.TSNE
- sklearn.metrics (fbeta_score, accuracy_score, precision_score, recall_score)

### Functions

#### `analyze_feature_importance(model, X_test, y_test, n_repeats=30, random_state=42)`

**Purpose:** Perform permutation-based feature importance analysis

**Parameters:**
- `model` (object): Trained model with `predict()` method
- `X_test` (ndarray): Test features, shape (n_samples, n_features)
- `y_test` (ndarray): Test labels, shape (n_samples,)
- `n_repeats` (int, default=30): Number of permutation repeats per feature
- `random_state` (int, default=42): Random seed for reproducibility

**Returns:**
- `dict`: Feature importance results containing:
  - `'importances_mean'`: Mean importance per feature
  - `'importances_std'`: Standard deviation per feature
  - `'critical_components'`: Indices of features with importance > 0.01
  - `'top_10_indices'`: Indices of top 10 most important features

**Side Effects:**
- Prints progress to stdout
- Generates matplotlib figure (2 subplots: all components, top 10)
- Saves figure as `feature_importance_analysis.png`

**Algorithm:**
1. Compute baseline F2 score on original test data
2. For each feature:
   - Permute feature values n_repeats times
   - Compute F2 score on permuted data
   - Importance = baseline_score - permuted_score
3. Identify critical components (importance > 0.01)
4. Visualize results

---

#### `visualize_feature_space_tsne(X_train, y_train, X_test, y_test, perplexity=30, random_state=42)`

**Purpose:** Create t-SNE 2D visualization of feature space

**Parameters:**
- `X_train` (ndarray): Training features, shape (n_samples, n_features)
- `y_train` (ndarray): Training labels, shape (n_samples,)
- `X_test` (ndarray): Test features, shape (n_samples, n_features)
- `y_test` (ndarray): Test labels, shape (n_samples,)
- `perplexity` (int, default=30): t-SNE perplexity parameter
- `random_state` (int, default=42): Random seed

**Returns:**
- `dict`: t-SNE analysis results containing:
  - `'tsne_coords'`: 2D t-SNE coordinates
  - `'centroid_distance'`: Euclidean distance between class centroids

**Side Effects:**
- Prints centroid distance and overlap analysis
- Generates matplotlib figure (3 subplots: by labels, by split, combined)
- Saves figure as `tsne_visualization.png`

**Algorithm:**
1. Combine train and test data
2. Apply t-SNE dimensionality reduction (n_features → 2D)
3. Calculate centroids for each class
4. Compute Euclidean distance between centroids
5. Determine if classes overlap (distance < 10 threshold)

---

#### `create_ensemble_predictions(fold_models, X_test, y_test)`

**Purpose:** Combine predictions from multiple fold models via averaging

**Parameters:**
- `fold_models` (list): List of trained models from cross-validation
- `X_test` (ndarray): Test features
- `y_test` (ndarray): Test labels

**Returns:**
- `dict`: Ensemble evaluation results containing:
  - `'f2'`: F2 score of ensemble
  - `'accuracy'`: Accuracy of ensemble
  - `'precision'`: Precision of ensemble
  - `'recall'`: Recall of ensemble

**Side Effects:**
- Prints ensemble performance metrics
- Prints comparison to individual folds

**Algorithm:**
1. Collect predictions from all fold models
2. Average predictions across folds
3. Apply threshold (> 0.5) to get binary predictions
4. Compute metrics on ensemble predictions

---

#### `test_noise_robustness(model, X_test, y_test, noise_levels=None)`

**Purpose:** Test model performance under Gaussian noise perturbations

**Parameters:**
- `model` (object): Trained model with `predict()` method
- `X_test` (ndarray): Test features
- `y_test` (ndarray): Test labels
- `noise_levels` (list, optional): Noise standard deviations to test
  - Default: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

**Returns:**
- `dict`: Noise robustness results containing:
  - `'noise_levels'`: List of noise levels tested
  - `'f2_scores'`: F2 scores at each noise level
  - `'accuracies'`: Accuracies at each noise level
  - `'precisions'`: Precisions at each noise level
  - `'recalls'`: Recalls at each noise level
  - `'robustness_score'`: F2 @ 10% noise / baseline F2
  - `'status'`: "PASS" if F2 > 0.8 at 10% noise, else "FAIL"

**Side Effects:**
- Prints robustness metrics table
- Prints PASS/FAIL status
- Generates matplotlib figure (4 subplots: F2, precision, recall, accuracy vs noise)
- Saves figure as `noise_robustness_test.png`

**Algorithm:**
1. Compute baseline metrics (no noise)
2. For each noise level:
   - Add Gaussian noise: X_noisy = X_test + N(0, noise_std)
   - Predict on noisy data
   - Compute all metrics
3. Calculate robustness score: F2(10% noise) / F2(0%)
4. Determine pass/fail (threshold: F2 > 0.8 at 10% noise)

---

#### `run_all_validation_analyses(model, fold_models, X_train, y_train, X_test, y_test)`

**Purpose:** Run all 4 validation analyses in sequence

**Parameters:**
- `model` (object): Best model for feature importance and noise testing
- `fold_models` (list): All fold models for ensemble evaluation
- `X_train`, `y_train`: Training data for t-SNE
- `X_test`, `y_test`: Test data

**Returns:**
- `dict`: Combined results from all analyses

**Side Effects:**
- Calls all 4 analysis functions
- Generates 3 PNG files
- Prints comprehensive validation summary

**Usage:**
```python
results = run_all_validation_analyses(
    model=best_model,
    fold_models=all_fold_models,
    X_train=training_features,
    y_train=training_labels,
    X_test=test_features,
    y_test=test_labels
)
```

---

## Module: run_validation_analysis.py

**Purpose:** Standalone script to execute full validation pipeline

**Imports:**
- Same as advanced_validation_analysis.py
- Additional: torch, torch.nn (for model training)

### Classes

#### `RegularizedMLP(nn.Module)`

**Purpose:** PyTorch neural network with strong regularization

**Constructor:**
```python
RegularizedMLP(input_size=50, hidden_size=64, dropout_rate=0.70)
```

**Parameters:**
- `input_size` (int, default=50): Number of input features (PCA components)
- `hidden_size` (int, default=64): Hidden layer size
- `dropout_rate` (float, default=0.70): Dropout probability

**Architecture:**
```
Input (50) → Linear → Dropout(0.70) → ReLU
           → Linear → Sigmoid → Output (1)
```

**Methods:**
- `forward(x)`: Forward pass returning probabilities [0, 1]

---

#### `SklearnWrapper`

**Purpose:** Wrap PyTorch model to provide sklearn-compatible interface

**Constructor:**
```python
SklearnWrapper(model)
```

**Parameters:**
- `model` (nn.Module): PyTorch model to wrap

**Methods:**
- `predict(X)`: Returns binary predictions (0 or 1)
- `predict_proba(X)`: Returns probability estimates [[p(0), p(1)], ...]

**Usage:**
```python
pytorch_model = RegularizedMLP()
sklearn_model = SklearnWrapper(pytorch_model)
predictions = sklearn_model.predict(X_test)
```

---

### Main Execution (if __name__ == "__main__")

**Steps:**
1. Load features from `features/features_pca_50.npy` and `features/weak_labels.csv`
2. Filter to labeled samples (true_label != -1)
3. Split 80/20 (80 training pool, 20 test)
4. Train RegularizedMLP for 50 epochs
5. Run all 3 validation analyses
6. Save results to JSON and PNG files

**Outputs:**
- `feature_importance_analysis.png`
- `tsne_visualization.png`
- `noise_robustness_test.png`
- `validation_analysis_results.json`

**Runtime:** ~10-15 minutes

---

## Notebooks (Entry Points)

### 1_feature_extraction.ipynb

**Purpose:** Extract deep features from MRI images using pre-trained ResNet50

**Key Functions (Inline):**
- `extract_features()`: Batch feature extraction from image directory
- `apply_pca()`: PCA dimensionality reduction

**Outputs:**
- `features/resnet50_features.npy`: Raw 2048D features
- `features/features_pca_50.npy`: PCA-reduced 50D features
- `features/labels.npy`: Ground truth labels
- `features/metadata.csv`: Image metadata (paths, splits, labels)

---

### 2_unsupervised_analysis.ipynb

**Purpose:** Generate weak labels via K-means clustering

**Key Classes (Inline):**
- None (uses sklearn.cluster.KMeans directly)

**Key Steps:**
1. Load PCA features
2. K-means clustering (k=2) on unlabeled data
3. Assign weak labels based on cluster membership
4. Filter by confidence (distance to centroid)
5. Save weak labels with confidence scores

**Outputs:**
- `features/weak_labels.csv`
- `features/weak_labels_filtered.csv`
- `features/weak_labels_high_confidence.csv`
- `features/clustering_summary.json`

---

### 3_semi_supervised_learning.ipynb

**Purpose:** Train and evaluate semi-supervised learning models

**Key Classes (Inline):**
```python
class RegularizedMLP(nn.Module):
    # Same as run_validation_analysis.py

class FeatureDataset(torch.utils.data.Dataset):
    # PyTorch dataset wrapper for numpy features

class EarlyStopping:
    # Early stopping callback (patience-based)
```

**Key Functions (Inline):**
- `train_model()`: Training loop with gradient clipping
- `evaluate_model()`: Compute metrics on test set
- `bootstrap_confidence_intervals()`: Non-parametric CI estimation
- `statistical_significance_test()`: Paired t-test between scenarios

**Outputs:**
- `detailed_cv_results.json`: Fold-by-fold results
- `scenario_comparison.csv`: Summary statistics
- In-memory trained models (not persisted)

---

## Public APIs (Exported Functions)

**From advanced_validation_analysis.py:**
```python
analyze_feature_importance()
visualize_feature_space_tsne()
create_ensemble_predictions()
test_noise_robustness()
run_all_validation_analyses()
```

**From run_validation_analysis.py:**
```python
RegularizedMLP (class)
SklearnWrapper (class)
```

**From notebooks:**
- No explicit exports (inline execution only)

---

## Known Gaps (Undocumented in Code)

1. **No function-level docstrings in notebooks:**
   - Inline functions in notebooks lack formal documentation
   - Behavior must be inferred from code

2. **No type hints:**
   - None of the functions use Python type hints
   - Parameter types inferred from usage

3. **No error handling documentation:**
   - Functions don't document exceptions they may raise
   - Error behavior must be tested empirically

4. **No performance guarantees:**
   - No documented time/space complexity
   - Runtime varies with input size

5. **No versioning:**
   - API stability not documented
   - Breaking changes not tracked

---

**Source:** This documentation is derived from:
- Function signatures and docstrings in code
- Code behavior analysis
- Parameter usage inspection
- Output observations from execution
