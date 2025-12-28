"""
Standalone script to run all validation analyses.
Executes: Feature Importance, t-SNE, and Noise Robustness tests.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import json

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("VALIDATION ANALYSIS - STANDALONE EXECUTION")
print("=" * 80)

# Load features (same way as notebook)
print("\nLoading features...")
import pandas as pd

features_pca_50 = np.load('features/features_pca_50.npy')
weak_labels_df = pd.read_csv('features/weak_labels.csv')

# Get labeled samples only (same as notebook)
labeled_df = weak_labels_df[weak_labels_df['true_label'] != -1].copy()
labeled_indices = labeled_df.index.tolist()
labeled_features = features_pca_50[labeled_indices]
labeled_labels = labeled_df['true_label'].values

print(f"\nLabeled samples: {len(labeled_labels)}")
print(f"  - Normal: {(labeled_labels == 0).sum()}")
print(f"  - Cancer: {(labeled_labels == 1).sum()}")

# 80/20 split
training_pool_pca, test_pca, training_pool_labels, test_labels = train_test_split(
    labeled_features,
    labeled_labels,
    test_size=0.20,
    stratify=labeled_labels,
    random_state=42
)

print(f"Training pool: {len(training_pool_pca)} samples")
print(f"Test set: {len(test_pca)} samples")

# ============================================================================
# STEP 1: TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: TRAINING MODEL")
print("=" * 80)

class RegularizedMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=64, dropout_rate=0.70):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = RegularizedMLP(input_size=50, hidden_size=64, dropout_rate=0.70)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(training_pool_pca),
    torch.FloatTensor(training_pool_labels).unsqueeze(1)
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

model.train()
for epoch in range(50):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

model.eval()
print("Model trained successfully.")

# Create wrapper
class SklearnWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, X):
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X))
            return (outputs.numpy() > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X)).numpy()
            return np.hstack([1 - outputs, outputs])

wrapped_model = SklearnWrapper(model)
print("Model wrapper created.")

# ============================================================================
# STEP 2: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

np.random.seed(42)

baseline_preds = wrapped_model.predict(test_pca)
baseline_score = fbeta_score(test_labels, baseline_preds, beta=2)

print(f"\nBaseline F2 Score: {baseline_score:.4f}")
print(f"Running 30 permutations for {test_pca.shape[1]} features...")

n_features = test_pca.shape[1]
n_repeats = 30
importances = np.zeros((n_features, n_repeats))

for feature_idx in range(n_features):
    for repeat in range(n_repeats):
        X_permuted = test_pca.copy()
        X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

        permuted_preds = wrapped_model.predict(X_permuted)
        permuted_score = fbeta_score(test_labels, permuted_preds, beta=2)

        importances[feature_idx, repeat] = baseline_score - permuted_score

    if (feature_idx + 1) % 10 == 0:
        print(f"  Processed {feature_idx + 1}/{n_features} features...")

importances_mean = importances.mean(axis=1)
importances_std = importances.std(axis=1)

critical_threshold = 0.01
critical_components = np.where(importances_mean > critical_threshold)[0]

print(f"\nTotal PCA components: {len(importances_mean)}")
print(f"Critical components (importance > {critical_threshold}): {len(critical_components)}")
print(f"Critical component indices: {critical_components.tolist()}")

top_10_indices = np.argsort(importances_mean)[-10:][::-1]
print(f"\nTop 10 most important components:")
for i, idx in enumerate(top_10_indices, 1):
    print(f"  {i}. Component {idx}: {importances_mean[idx]:.4f} +/- {importances_std[idx]:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(range(len(importances_mean)), importances_mean,
            yerr=importances_std, alpha=0.7, capsize=2)
axes[0].axhline(y=critical_threshold, color='r', linestyle='--',
                label=f'Threshold ({critical_threshold})')
axes[0].set_xlabel('PCA Component Index')
axes[0].set_ylabel('Permutation Importance')
axes[0].set_title('Feature Importance: All Components')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].barh(range(10), importances_mean[top_10_indices],
             xerr=importances_std[top_10_indices], alpha=0.7, capsize=3)
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([f'PC{idx}' for idx in top_10_indices])
axes[1].set_xlabel('Permutation Importance')
axes[1].set_title('Top 10 Most Important Components')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved: feature_importance_analysis.png")
plt.close()

# ============================================================================
# STEP 3: t-SNE VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: t-SNE VISUALIZATION")
print("=" * 80)

X_combined = np.vstack([training_pool_pca, test_pca])
y_combined = np.concatenate([training_pool_labels, test_labels])
split_labels = ['Train'] * len(training_pool_pca) + ['Test'] * len(test_pca)

print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_combined)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1
for label in [0, 1]:
    mask = y_combined == label
    label_name = 'Normal' if label == 0 else 'Cancer'
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=label_name, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].set_title('t-SNE: Colored by True Labels')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2
for split in ['Train', 'Test']:
    mask = np.array(split_labels) == split
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=split, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].set_title('t-SNE: Colored by Train/Test Split')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3
colors = {'Train_Normal': 'blue', 'Train_Cancer': 'red',
          'Test_Normal': 'lightblue', 'Test_Cancer': 'pink'}
for split in ['Train', 'Test']:
    for label in [0, 1]:
        split_mask = np.array(split_labels) == split
        label_mask = y_combined == label
        mask = split_mask & label_mask
        label_name = 'Normal' if label == 0 else 'Cancer'
        color_key = f'{split}_{label_name}'
        axes[2].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       label=f'{split} - {label_name}',
                       color=colors[color_key], alpha=0.7, s=100,
                       edgecolors='black', linewidth=0.5)
axes[2].set_xlabel('t-SNE Dimension 1')
axes[2].set_ylabel('t-SNE Dimension 2')
axes[2].set_title('t-SNE: Combined View')
axes[2].legend(loc='best', fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
print("\nSaved: tsne_visualization.png")
plt.close()

train_normal = X_tsne[(np.array(split_labels) == 'Train') & (y_combined == 0)]
train_cancer = X_tsne[(np.array(split_labels) == 'Train') & (y_combined == 1)]

centroid_normal = train_normal.mean(axis=0)
centroid_cancer = train_cancer.mean(axis=0)
centroid_distance = np.linalg.norm(centroid_normal - centroid_cancer)

print(f"\nClass Separation Metrics:")
print(f"  Distance between centroids: {centroid_distance:.2f}")
print(f"  Normal cluster std: {train_normal.std(axis=0).mean():.2f}")
print(f"  Cancer cluster std: {train_cancer.std(axis=0).mean():.2f}")

if centroid_distance > 10:
    print("  -> Classes are well-separated (explains high accuracy)")
else:
    print("  -> Classes overlap (model found subtle patterns)")

# ============================================================================
# STEP 4: NOISE ROBUSTNESS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: NOISE ROBUSTNESS TEST")
print("=" * 80)

noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results = {'noise_levels': [], 'f2_scores': [], 'precision_scores': [],
           'recall_scores': [], 'accuracy_scores': []}

print(f"\nTesting noise levels: {noise_levels}")
print(f"{'Noise':<10} {'F2':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10}")
print("-" * 60)

for noise_std in noise_levels:
    if noise_std > 0:
        noise = np.random.randn(*test_pca.shape) * noise_std
        X_noisy = test_pca + noise
    else:
        X_noisy = test_pca

    y_pred = wrapped_model.predict(X_noisy)

    f2 = fbeta_score(test_labels, y_pred, beta=2)
    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall = recall_score(test_labels, y_pred)
    accuracy = accuracy_score(test_labels, y_pred)

    results['noise_levels'].append(noise_std)
    results['f2_scores'].append(f2)
    results['precision_scores'].append(precision)
    results['recall_scores'].append(recall)
    results['accuracy_scores'].append(accuracy)

    print(f"{noise_std:<10.2f} {f2:<10.4f} {precision:<12.4f} {recall:<10.4f} {accuracy:<10.4f}")

baseline_f2 = results['f2_scores'][0]
f2_at_10pct = results['f2_scores'][2]
robustness_score = f2_at_10pct / baseline_f2

print(f"\nRobustness Metrics:")
print(f"  Baseline F2: {baseline_f2:.4f}")
print(f"  F2 at 10% noise: {f2_at_10pct:.4f}")
print(f"  Robustness score: {robustness_score:.4f}")
print(f"  Performance drop: {(1-robustness_score)*100:.2f}%")

target_f2 = 0.80
if f2_at_10pct > target_f2:
    print(f"  Status: PASS (F2 > {target_f2})")
else:
    print(f"  Status: FAIL (F2 < {target_f2})")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('F2 Score', results['f2_scores'], axes[0, 0]),
    ('Precision', results['precision_scores'], axes[0, 1]),
    ('Recall', results['recall_scores'], axes[1, 0]),
    ('Accuracy', results['accuracy_scores'], axes[1, 1])
]

for metric_name, scores, ax in metrics:
    ax.plot(results['noise_levels'], scores, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=target_f2, color='r', linestyle='--', label=f'Target ({target_f2})')
    ax.axvline(x=0.10, color='g', linestyle='--', alpha=0.5, label='10% Noise')
    ax.set_xlabel('Noise Level (Std Dev)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs. Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('noise_robustness_test.png', dpi=300, bbox_inches='tight')
print("\nSaved: noise_robustness_test.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

validation_results = {
    'feature_importance': {
        'n_critical_components': int(len(critical_components)),
        'critical_indices': critical_components.tolist(),
        'top_10_indices': top_10_indices.tolist(),
        'baseline_f2': float(baseline_score)
    },
    'tsne': {
        'centroid_distance': float(centroid_distance)
    },
    'robustness': {
        'baseline_f2': float(baseline_f2),
        'f2_at_10pct_noise': float(f2_at_10pct),
        'robustness_score': float(robustness_score),
        'noise_levels': noise_levels,
        'f2_scores': results['f2_scores']
    }
}

with open('validation_analysis_results.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

print("\nSaved: validation_analysis_results.json")

print("\n" + "=" * 80)
print("VALIDATION ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - feature_importance_analysis.png")
print("  - tsne_visualization.png")
print("  - noise_robustness_test.png")
print("  - validation_analysis_results.json")
print("\nAll validation tests completed successfully!")
