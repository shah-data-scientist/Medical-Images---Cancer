# Alternative Validation Plan
## (When External Data is Not Available)

**Date**: 2025-12-26
**Context**: No external validation dataset available
**Goal**: Maximize confidence in model performance using only internal data

---

## 🎯 Strategy 1: Optimize Data Split (IMMEDIATE)

### Current Situation
- **Total labeled**: 100 samples (50 normal, 50 cancer)
- **Current split**: 70 training / 30 test
- **Actual training**: 59 samples (11 used for validation in CV)
- **Problem**: Tiny training set → high overfitting risk

### Proposed New Split

**Option A: 80/20 Split (Recommended)**
```
Training: 80 samples (40 normal, 40 cancer)
Test: 20 samples (10 normal, 10 cancer)

Benefits:
✅ +36% more training data (59 → 80 samples)
✅ Still statistically valid test set (20 samples)
✅ Better parameter-to-sample ratio (42:1 instead of 56:1)
✅ Can still run 5-fold CV on 80 samples

Trade-offs:
⚠️ Smaller test set → wider confidence intervals
⚠️ But 20 samples still sufficient for binomial test
```

**Option B: Leave-One-Out Cross-Validation (LOOCV)**
```
Training: 99 samples
Test: 1 sample (repeated 100 times)

Benefits:
✅ Maximum training data usage
✅ 100 independent test results
✅ Very thorough validation

Trade-offs:
⚠️ Computationally expensive (100 model trainings)
⚠️ High variance in individual predictions
⚠️ Takes ~2-3 hours to complete
```

**Recommendation**: Start with Option A (80/20), then try LOOCV if needed

---

## 🎯 Strategy 2: Feature Analysis & Interpretability

### Goal
Understand **why** scores are so high (89-99%) to determine if it's real or artifact

### Actions

#### 2.1 Feature Importance via Permutation
```python
from sklearn.inspection import permutation_importance

# Train model
model = train_final_model()

# Permutation importance
perm_importance = permutation_importance(
    model, test_features, test_labels,
    n_repeats=30, random_state=42
)

# Identify critical PCA components
critical_components = np.where(perm_importance.importances_mean > 0.01)[0]
print(f"Critical components: {critical_components}")
```

**Expected Outcome**:
- If only 5-10 components matter → features are highly discriminative
- If all 50 matter → need full feature set

#### 2.2 Ablation Study
Remove PCA components systematically and measure performance drop

```python
results = []
for n_components in [50, 40, 30, 20, 10, 5]:
    features_subset = features_pca[:, :n_components]
    model = train_model(features_subset)
    f2 = evaluate(model)
    results.append((n_components, f2))

# Plot performance vs. components
plt.plot(n_components, f2_scores)
```

**Expected Outcome**:
- If F2 stays high even with 10 components → task is genuinely easy
- If F2 drops sharply → need all 50 components

#### 2.3 Visualize Decision Boundary
```python
from sklearn.manifold import TSNE
import umap

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features_pca_50)

# Plot with true labels
plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1],
            label='Normal', alpha=0.6)
plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1],
            label='Cancer', alpha=0.6)
plt.legend()
plt.title('t-SNE Visualization of PCA Features')
```

**Expected Outcome**:
- If classes are clearly separated → explains high accuracy
- If mixed → suggests model found subtle patterns

---

## 🎯 Strategy 3: Robustness Testing

### 3.1 Noise Injection Test
Add controlled noise to test set and measure performance degradation

```python
noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
results = []

for noise_std in noise_levels:
    # Add Gaussian noise to features
    noisy_features = test_features + np.random.randn(*test_features.shape) * noise_std

    # Evaluate
    f2 = evaluate_model(model, noisy_features, test_labels)
    results.append((noise_std, f2))

# Plot robustness curve
plt.plot(noise_levels, f2_scores)
plt.xlabel('Noise Level (Std)')
plt.ylabel('F2 Score')
plt.title('Model Robustness to Feature Noise')
```

**Interpretation**:
- Robust model: Gradual degradation with noise
- Overfitted model: Sharp drop with small noise
- **Target**: F2 should stay >80% with 10% noise

### 3.2 Feature Dropout Test
Randomly drop features and measure performance

```python
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
results = []

for dropout_rate in dropout_rates:
    # Randomly zero out features
    mask = np.random.rand(*test_features.shape) > dropout_rate
    dropped_features = test_features * mask

    f2 = evaluate_model(model, dropped_features, test_labels)
    results.append((dropout_rate, f2))
```

**Interpretation**:
- If F2 drops rapidly → model relies on all features
- If F2 stays high → model uses redundant features (good!)

---

## 🎯 Strategy 4: Learning Curves

### Goal
Determine if more training data would help

```python
from sklearn.model_selection import learning_curve

train_sizes = [10, 20, 30, 40, 50, 59]
train_scores, val_scores = [], []

for train_size in train_sizes:
    # Use first `train_size` samples
    subset_features = all_labeled_pca[:train_size]
    subset_labels = all_labeled_labels[:train_size]

    # Train and evaluate
    model = train_model(subset_features, subset_labels)
    train_f2 = evaluate(model, subset_features, subset_labels)
    val_f2 = evaluate(model, test_features, test_labels)

    train_scores.append(train_f2)
    val_scores.append(val_f2)

# Plot learning curves
plt.plot(train_sizes, train_scores, label='Train')
plt.plot(train_sizes, val_scores, label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('F2 Score')
plt.legend()
```

**Interpretation**:
- **Converging curves** → model has enough data
- **Diverging curves** → overfitting, need more data
- **Rising validation curve** → more data would help

**Expected with 59 samples**: Likely diverging → confirms need for more data

---

## 🎯 Strategy 5: Ensemble & Confidence

### 5.1 Use All 5 Fold Models (Ensemble)
```python
# Instead of picking best fold, use ALL 5 models
all_predictions = []

for fold_model in [model_fold1, model_fold2, model_fold3, model_fold4, model_fold5]:
    probs = fold_model.predict_proba(test_features)
    all_predictions.append(probs)

# Average predictions
ensemble_probs = np.mean(all_predictions, axis=0)
ensemble_preds = (ensemble_probs[:, 1] > 0.5).astype(int)

# Evaluate ensemble
f2_ensemble = fbeta_score(test_labels, ensemble_preds, beta=2)
```

**Expected**: +2-5% improvement over single model

### 5.2 Prediction Confidence Analysis
```python
# Analyze prediction confidence
high_confidence = ensemble_probs.max(axis=1) > 0.9
low_confidence = ensemble_probs.max(axis=1) < 0.7

print(f"High confidence predictions: {high_confidence.sum()}")
print(f"Low confidence predictions: {low_confidence.sum()}")

# Check accuracy by confidence
high_conf_accuracy = accuracy_score(
    test_labels[high_confidence],
    ensemble_preds[high_confidence]
)
low_conf_accuracy = accuracy_score(
    test_labels[low_confidence],
    ensemble_preds[low_confidence]
)

print(f"High confidence accuracy: {high_conf_accuracy:.2%}")
print(f"Low confidence accuracy: {low_conf_accuracy:.2%}")
```

**Interpretation**:
- If high-confidence predictions are accurate → model is well-calibrated
- If low-confidence predictions fail → model knows when it's uncertain (good!)

---

## 🎯 Strategy 6: Statistical Validation

### 6.1 Permutation Test (Verify Performance Isn't Random)
```python
from sklearn.utils import shuffle

# Run permutation test
n_permutations = 1000
permutation_scores = []

for _ in range(n_permutations):
    # Shuffle labels randomly
    shuffled_labels = shuffle(test_labels, random_state=None)

    # Evaluate with shuffled labels
    f2_perm = fbeta_score(shuffled_labels, predictions, beta=2)
    permutation_scores.append(f2_perm)

# Calculate p-value
actual_f2 = 0.989  # Your actual score
p_value = (np.array(permutation_scores) >= actual_f2).mean()

print(f"Permutation test p-value: {p_value}")
print(f"Performance is {'random' if p_value > 0.05 else 'statistically significant'}")
```

**Expected**: p < 0.001 (performance is NOT random)

### 6.2 Binomial Test (Small Test Set Validity)
```python
from scipy.stats import binomtest

# With 30 test samples, what's the probability of 29/30 correct by chance?
result = binomtest(
    k=29,  # Correct predictions (assuming 1 error)
    n=30,  # Total samples
    p=0.5,  # Null hypothesis (random guessing)
    alternative='greater'
)

print(f"Binomial test p-value: {result.pvalue}")
```

**Expected**: p < 0.001 (29/30 is NOT by chance)

---

## 🎯 Strategy 7: Error Analysis

### Analyze the FEW mistakes the model makes

```python
# Identify misclassified samples
errors = predictions != test_labels
error_indices = np.where(errors)[0]

print(f"Total errors: {errors.sum()} out of {len(test_labels)}")

if errors.sum() > 0:
    # Analyze error characteristics
    for idx in error_indices:
        true_label = test_labels[idx]
        pred_label = predictions[idx]
        pred_prob = probabilities[idx]

        print(f"Sample {idx}:")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print(f"  Confidence: {pred_prob:.2%}")

        # Check if this sample is an outlier
        # (distance to nearest neighbor)
        distances = np.linalg.norm(test_features - test_features[idx], axis=1)
        nearest_dist = np.partition(distances, 1)[1]  # 2nd closest (1st is itself)
        print(f"  Distance to nearest neighbor: {nearest_dist:.3f}")
```

**Questions to Answer**:
1. Are errors on edge cases (low confidence)?
2. Are errors clustered (systematic issue)?
3. Are errors outliers (unusual samples)?

---

## 📊 Recommended Execution Order

### Week 1: Quick Wins
1. ✅ **Expand to 80/20 split** (immediate +36% training data)
2. ✅ **Feature importance analysis** (understand what matters)
3. ✅ **t-SNE visualization** (see if classes are separable)
4. ✅ **Ensemble 5-fold models** (easy +2-5% improvement)

### Week 2: Robustness
5. ✅ **Noise injection test** (verify robustness)
6. ✅ **Learning curves** (confirm need for more data)
7. ✅ **Permutation test** (prove performance isn't random)

### Week 3: Deep Dive
8. ✅ **Ablation study** (optimal number of PCA components)
9. ✅ **Error analysis** (understand rare failures)
10. ✅ **LOOCV** (if needed for maximum validation)

---

## 🎯 Expected Outcomes

### If Features Are Genuinely Powerful
- t-SNE shows clear separation
- Ablation shows 10-15 components sufficient
- Noise injection shows gradual degradation
- Performance is real, just easier task than expected

### If Model is Overfitting
- t-SNE shows mixed classes
- Ablation shows need for all 50 components
- Noise injection shows sharp performance drop
- Need more training data urgently

### Most Likely Scenario
**Features are high-quality BUT task is dataset-specific**
- Classes are separable in this dataset
- Performance will drop on real-world deployment
- Need 200-500 samples to handle population diversity

---

## 💰 Cost-Benefit Analysis

### No Cost (Use Existing Data)
- 80/20 split: **Free, +36% training data**
- Feature analysis: **Free, deep insights**
- Ensemble: **Free, +2-5% performance**
- Robustness testing: **Free, confidence boost**

### Low Cost (Label More Data)
- 100 → 200 samples: €300 (already labeled 100, need 100 more)
- 100 → 500 samples: €1,200 (need 400 more)
- **Recommended**: Label 200 more (€600) for 3x data increase

### High Cost (Collect New Data)
- External dataset: Time-consuming, may not exist
- New patient recruitment: €€€, months of time
- **NOT RECOMMENDED**: Focus on maximizing current data first

---

## ✅ Action Plan Summary

**This Week (No Cost)**:
1. Implement 80/20 split → Re-run experiments
2. Feature importance → Identify critical components
3. t-SNE visualization → Understand separability
4. Ensemble models → Boost performance
5. Noise injection → Test robustness

**Next Week (Low Cost)**:
6. Learning curves → Confirm data needs
7. Budget approval for 200 more labels (€600)
8. Label 100 more samples → 200 total training

**Long-term (Contingent)**:
9. If budget allows: Label 500 total (€1,500)
10. Deploy with monitoring and drift detection

---

**Status**: Ready to implement
**First Action**: Modify data split to 80/20 and re-run all experiments
