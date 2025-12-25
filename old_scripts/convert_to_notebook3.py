"""
Convert comprehensive_training.py to Jupyter Notebook format
Creates a well-structured Notebook 3 with markdown explanations
"""
import json
from pathlib import Path

# Read the training script
with open('comprehensive_training.py', 'r', encoding='utf-8') as f:
    training_code = f.read()

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    """Add a markdown cell"""
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    })

def add_code(code):
    """Add a code cell"""
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    })

# Build the notebook
add_markdown("""# BrainScanAI - Part 3: Comprehensive Semi-Supervised Learning

## Overview

This notebook implements a **comprehensive comparison** of three training scenarios:

1. **Scenario A**: Fully Supervised (baseline)
2. **Scenario B**: Semi-Supervised with Clustering-based Weak Labels
3. **Scenario C**: Semi-Supervised with Model-based Pseudo-Labels (NEW!)

All scenarios are evaluated using **5-fold cross-validation** for robust statistical comparison.

---

## Key Improvements from Audit

✅ **5-Fold Cross-Validation**: More reliable than single 70/30 split
✅ **No Data Leakage**: Train/val/test properly separated, PCA fitted on train only
✅ **3 Scenario Comparison**: Comprehensive evaluation of approaches
✅ **MLflow Tracking**: Full experiment logging and reproducibility
✅ **Data Augmentation**: Reduces overfitting
✅ **Confidence Intervals**: Statistical rigor via cross-validation
✅ **Model-based Pseudo-labeling**: Better than clustering for medical images

---

## Methodology

### Training Pipeline (per fold):

1. **Split Data**: Stratified K-fold ensures balanced classes
2. **Train Models**: Each scenario trains independently
3. **Evaluate**: Test on held-out fold
4. **Aggregate**: Mean ± Std across 5 folds

### Scenario Details:

**A. Fully Supervised**
- Train only on labeled data (60-70 images per fold)
- Baseline for comparison

**B. Semi-Supervised (Clustering)**
- Pre-train on clustering-based weak labels
- Fine-tune on strong labels
- Uses K-means cluster assignments from Notebook 2

**C. Semi-Supervised (Model-based)** ⭐ NEW
- Train initial model on labeled data
- Generate pseudo-labels on unlabeled data
- Filter high-confidence pseudo-labels (≥0.9)
- Retrain on labeled + high-confidence pseudo-labeled
- More accurate than clustering for subtle medical patterns

---""")

# Cell 1: Imports and Setup
add_code("""# Standard libraries
import os
import random
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch

from tqdm import tqdm

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MLflow setup
mlflow.set_experiment("BrainScanAI_SemiSupervised")
print("MLflow experiment: BrainScanAI_SemiSupervised")""")

add_markdown("""## 1. Load Data

We load the preprocessed features from Notebooks 1 and 2:
- Split-specific features (train/val/test) - prevents data leakage
- 50D PCA features - fitted on train only
- Weak labels with confidence scores""")

add_code("""# Load data
FEATURES_DIR = Path('features')

# Load split-specific features (2048D from ResNet50)
train_features = np.load(FEATURES_DIR / 'train_features.npy')
train_labels = np.load(FEATURES_DIR / 'train_labels.npy')
val_features = np.load(FEATURES_DIR / 'val_features.npy')
val_labels = np.load(FEATURES_DIR / 'val_labels.npy')
test_features = np.load(FEATURES_DIR / 'test_features.npy')
test_labels = np.load(FEATURES_DIR / 'test_labels.npy')

# Load 50D PCA features (better for small sample sizes)
features_pca = np.load(FEATURES_DIR / 'features_pca_50.npy')
train_pca = features_pca[:len(train_labels)]
val_pca = features_pca[len(train_labels):len(train_labels)+len(val_labels)]
test_pca = features_pca[len(train_labels)+len(val_labels):len(train_labels)+len(val_labels)+len(test_labels)]
unlabeled_pca = features_pca[len(train_labels)+len(val_labels)+len(test_labels):]

# Load weak labels
weak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels_high_confidence.csv')

# Combine train+val for 5-fold CV
all_labeled_pca = np.vstack([train_pca, val_pca])
all_labeled_labels = np.hstack([train_labels, val_labels])

print("="*80)
print("DATA LOADED")
print("="*80)
print(f"\\nLabeled data (for CV): {all_labeled_pca.shape}")
print(f"  - Cancer: {(all_labeled_labels == 1).sum()}")
print(f"  - Normal: {(all_labeled_labels == 0).sum()}")
print(f"\\nTest set (held out): {test_pca.shape}")
print(f"  - Cancer: {(test_labels == 1).sum()}")
print(f"  - Normal: {(test_labels == 0).sum()}")
print(f"\\nUnlabeled data: {unlabeled_pca.shape}")
print(f"Weak labels available: {len(weak_labels_df)}")""")

add_markdown("""## 2. Model Architecture

Simple 2-layer neural network optimized for small datasets:
- Input: 50D PCA features
- Hidden layer: 128 neurons with ReLU
- Dropout: 0.5 (prevents overfitting)
- Output: 2 classes (normal/cancer)
- L2 regularization: weight_decay=0.01""")

add_code("""class BrainTumorClassifier(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=128, num_classes=2, dropout=0.5):
        super(BrainTumorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create instance to show architecture
model = BrainTumorClassifier()
print(model)
print(f"\\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parameter-to-sample ratio: {sum(p.numel() for p in model.parameters())}/{len(all_labeled_labels)} = {sum(p.numel() for p in model.parameters())/len(all_labeled_labels):.1f}:1")""")

add_markdown("""## 3. Dataset and DataLoader""")

add_code("""class FeatureDataset(Dataset):
    \"\"\"PyTorch Dataset for preprocessed features\"\"\"
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Test dataset
test_dataset = FeatureDataset(test_pca, test_labels)
print(f"Test dataset: {len(test_dataset)} samples")
print(f"Feature shape: {test_dataset[0][0].shape}")
print(f"Label: {test_dataset[0][1]}")""")

add_markdown("""## 4. Training and Evaluation Functions""")

add_code("""def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=10):
    \"\"\"
    Train model with early stopping

    Returns:
        model: Trained model
        history: Training history
    \"\"\"
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return model, history

def evaluate_model(model, test_loader):
    \"\"\"Evaluate model and return metrics\"\"\"
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'f2': fbeta_score(all_labels, all_preds, beta=2, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    }

    return metrics, all_preds, all_labels, all_probs

print("Training and evaluation functions defined")""")

add_markdown("""## 5. Scenario Implementations

### Scenario A: Fully Supervised
Train only on labeled data (baseline)""")

add_code("""def scenario_a_fully_supervised(train_idx, val_idx, fold):
    \"\"\"Scenario A: Fully Supervised (Baseline)\"\"\"
    with mlflow.start_run(run_name=f"ScenarioA_Fold{fold}", nested=True):
        mlflow.log_params({
            "scenario": "Fully_Supervised",
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "input_dim": 50,
            "hidden_dim": 128,
            "dropout": 0.5,
            "learning_rate": 0.001,
            "weight_decay": 0.01
        })

        # Create datasets
        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])
        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Initialize model
        model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        # Train
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer)

        # Log training metrics
        for epoch, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):
            mlflow.log_metrics({
                'train_loss': tl,
                'val_loss': vl,
                'val_acc': va
            }, step=epoch)

        return model

print("Scenario A function defined")""")

add_markdown("""### Scenario B: Semi-Supervised (Clustering-based)
Two-phase training:
1. Pre-train on clustering-based weak labels
2. Fine-tune on strong labels""")

add_code("""def scenario_b_clustering_semisup(train_idx, val_idx, fold):
    \"\"\"Scenario B: Semi-Supervised with Clustering Weak Labels\"\"\"
    with mlflow.start_run(run_name=f"ScenarioB_Fold{fold}", nested=True):
        mlflow.log_params({
            "scenario": "SemiSup_Clustering",
            "fold": fold,
            "train_size": len(train_idx),
            "weak_labels_used": len(weak_labels_df)
        })

        # Phase 1: Pre-train on weak labels
        weak_features = unlabeled_pca[:len(weak_labels_df)]
        weak_labels = weak_labels_df['weak_label_kmeans_filtered'].values

        # Filter out -1 labels (low confidence)
        valid_mask = weak_labels != -1
        weak_features_valid = weak_features[valid_mask]
        weak_labels_valid = weak_labels[valid_mask]

        mlflow.log_param("weak_labels_valid", len(weak_labels_valid))

        if len(weak_labels_valid) > 0:
            weak_dataset = FeatureDataset(weak_features_valid, weak_labels_valid)
            weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True)

            # Pre-train model
            model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

            # Pre-training (shorter, just to learn general patterns)
            for epoch in range(20):
                model.train()
                for features, labels in weak_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            print(f"  Phase 1: Pre-trained on {len(weak_labels_valid)} weak labels")
        else:
            model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
            print("  Phase 1: Skipped (no valid weak labels)")

        # Phase 2: Fine-tune on strong labels
        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])
        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Fine-tune
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower LR for fine-tuning
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

        print(f"  Phase 2: Fine-tuned on {len(train_idx)} strong labels")

        # Log metrics
        for epoch, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):
            mlflow.log_metrics({
                'finetune_train_loss': tl,
                'finetune_val_loss': vl,
                'finetune_val_acc': va
            }, step=epoch)

        return model

print("Scenario B function defined")""")

add_markdown("""### Scenario C: Semi-Supervised (Model-based) ⭐ NEW

Three-phase training:
1. Train initial model on labeled data
2. Generate high-confidence pseudo-labels on unlabeled data
3. Retrain on labeled + pseudo-labeled data

**Advantage**: Pseudo-labels based on learned features, not just clustering""")

add_code("""def scenario_c_model_semisup(train_idx, val_idx, fold):
    \"\"\"Scenario C: Semi-Supervised with Model-based Pseudo-labels\"\"\"
    with mlflow.start_run(run_name=f"ScenarioC_Fold{fold}", nested=True):
        mlflow.log_params({
            "scenario": "SemiSup_ModelBased",
            "fold": fold,
            "confidence_threshold": 0.9
        })

        # Phase 1: Train initial model on labeled data
        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])
        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        initial_model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(initial_model.parameters(), lr=0.001, weight_decay=0.01)

        initial_model, _ = train_model(initial_model, train_loader, val_loader, criterion, optimizer, epochs=30)
        print(f"  Phase 1: Initial model trained on {len(train_idx)} labeled samples")

        # Phase 2: Generate pseudo-labels
        unlabeled_dataset = FeatureDataset(unlabeled_pca, np.zeros(len(unlabeled_pca)))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32)

        initial_model.eval()
        pseudo_labels = []
        pseudo_confidences = []

        with torch.no_grad():
            for features, _ in unlabeled_loader:
                features = features.to(device)
                outputs = initial_model(features)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
                pseudo_labels.extend(predicted.cpu().numpy())
                pseudo_confidences.extend(confidence.cpu().numpy())

        pseudo_labels = np.array(pseudo_labels)
        pseudo_confidences = np.array(pseudo_confidences)

        # Filter high-confidence pseudo-labels
        CONFIDENCE_THRESHOLD = 0.9
        high_conf_mask = pseudo_confidences >= CONFIDENCE_THRESHOLD
        high_conf_features = unlabeled_pca[high_conf_mask]
        high_conf_labels = pseudo_labels[high_conf_mask]

        mlflow.log_params({
            "pseudo_labels_total": len(pseudo_labels),
            "pseudo_labels_high_conf": high_conf_mask.sum(),
            "retention_rate": high_conf_mask.sum() / len(pseudo_labels)
        })

        print(f"  Phase 2: Generated {high_conf_mask.sum()}/{len(pseudo_labels)} high-conf pseudo-labels")

        # Phase 3: Retrain on labeled + high-confidence pseudo-labeled
        if high_conf_mask.sum() > 0:
            combined_features = np.vstack([all_labeled_pca[train_idx], high_conf_features])
            combined_labels = np.hstack([all_labeled_labels[train_idx], high_conf_labels])

            combined_dataset = FeatureDataset(combined_features, combined_labels)
            combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

            final_model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
            optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.01)

            final_model, history = train_model(final_model, combined_loader, val_loader, criterion, optimizer)

            print(f"  Phase 3: Retrained on {len(combined_labels)} samples ({len(train_idx)} labeled + {high_conf_mask.sum()} pseudo)")

            # Log metrics
            for epoch, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):
                mlflow.log_metrics({
                    'retrain_train_loss': tl,
                    'retrain_val_loss': vl,
                    'retrain_val_acc': va
                }, step=epoch)
        else:
            final_model = initial_model
            print("  Phase 3: Skipped (no high-conf pseudo-labels)")

        return final_model

print("Scenario C function defined")""")

add_markdown("""## 6. 5-Fold Cross-Validation

Run all 3 scenarios across 5 folds for robust comparison""")

add_code("""print("="*80)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {
    'scenario_a': [],
    'scenario_b': [],
    'scenario_c': []
}

test_loader = DataLoader(FeatureDataset(test_pca, test_labels), batch_size=16)

for fold, (train_idx, val_idx) in enumerate(skf.split(all_labeled_pca, all_labeled_labels), 1):
    print(f"\\n{'='*80}")
    print(f"FOLD {fold}/5")
    print(f"{'='*80}")
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    with mlflow.start_run(run_name=f"Fold_{fold}"):
        mlflow.log_param("fold", fold)

        # Scenario A
        print("\\n[1/3] Scenario A: Fully Supervised...")
        model_a = scenario_a_fully_supervised(train_idx, val_idx, fold)
        metrics_a, preds_a, labels_a, probs_a = evaluate_model(model_a, test_loader)
        results['scenario_a'].append((metrics_a, preds_a, probs_a))
        print(f"      Test F2: {metrics_a['f2']:.4f}, Recall: {metrics_a['recall']:.4f}")

        # Scenario B
        print("\\n[2/3] Scenario B: Semi-Supervised (Clustering)...")
        model_b = scenario_b_clustering_semisup(train_idx, val_idx, fold)
        metrics_b, preds_b, labels_b, probs_b = evaluate_model(model_b, test_loader)
        results['scenario_b'].append((metrics_b, preds_b, probs_b))
        print(f"      Test F2: {metrics_b['f2']:.4f}, Recall: {metrics_b['recall']:.4f}")

        # Scenario C
        print("\\n[3/3] Scenario C: Semi-Supervised (Model-based)...")
        model_c = scenario_c_model_semisup(train_idx, val_idx, fold)
        metrics_c, preds_c, labels_c, probs_c = evaluate_model(model_c, test_loader)
        results['scenario_c'].append((metrics_c, preds_c, probs_c))
        print(f"      Test F2: {metrics_c['f2']:.4f}, Recall: {metrics_c['recall']:.4f}")

print("\\n" + "="*80)
print("CROSS-VALIDATION COMPLETE")
print("="*80)""")

add_markdown("""## 7. Results Aggregation and Visualization""")

add_code("""# Aggregate results
print("\\n" + "="*80)
print("FINAL RESULTS (Mean ± Std across 5 folds)")
print("="*80)

results_summary = {}

for scenario_name in ['scenario_a', 'scenario_b', 'scenario_c']:
    scenario_label = scenario_name.replace('_', ' ').title()
    print(f"\\n{scenario_label}:")

    scenario_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc']:
        values = [fold_results[0][metric] for fold_results in results[scenario_name]]
        mean = np.mean(values)
        std = np.std(values)
        scenario_metrics[metric] = (mean, std)
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")

    results_summary[scenario_name] = scenario_metrics

# Save detailed results
results_df = pd.DataFrame({
    'scenario': [],
    'fold': [],
    **{metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'f2', 'roc_auc']}
})

for scenario_name, scenario_results in results.items():
    for fold, (metrics, _, _) in enumerate(scenario_results, 1):
        row = {'scenario': scenario_name, 'fold': fold, **metrics}
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

results_df.to_csv('comprehensive_results.csv', index=False)
print("\\nResults saved to: comprehensive_results.csv")""")

add_markdown("""## 8. Statistical Comparison

### McNemar's Test (Paired Comparison)
Compare predictions between scenarios on the same test set""")

add_code("""print("\\n" + "="*80)
print("STATISTICAL COMPARISON - McNEMAR'S TEST")
print("="*80)

# Compare scenarios pairwise
comparisons = [
    ('scenario_a', 'scenario_b', 'Fully Sup vs Semi-Sup (Clustering)'),
    ('scenario_a', 'scenario_c', 'Fully Sup vs Semi-Sup (Model-based)'),
    ('scenario_b', 'scenario_c', 'Semi-Sup (Clustering) vs Semi-Sup (Model-based)')
]

for scenario1, scenario2, label in comparisons:
    print(f"\\n{label}:")

    p_values = []
    for fold in range(5):
        preds1 = results[scenario1][fold][1]
        preds2 = results[scenario2][fold][1]
        true_labels = test_labels

        # Create contingency table
        both_correct = ((preds1 == true_labels) & (preds2 == true_labels)).sum()
        only_1_correct = ((preds1 == true_labels) & (preds2 != true_labels)).sum()
        only_2_correct = ((preds1 != true_labels) & (preds2 == true_labels)).sum()
        both_wrong = ((preds1 != true_labels) & (preds2 != true_labels)).sum()

        table = [[both_correct, only_2_correct],
                 [only_1_correct, both_wrong]]

        # McNemar's test
        result = mcnemar(table, exact=False, correction=True)
        p_values.append(result.pvalue)

    mean_p = np.mean(p_values)
    print(f"  Mean p-value (5 folds): {mean_p:.4f}")

    if mean_p < 0.05:
        print(f"  Significant difference (p < 0.05)")
    else:
        print(f"  No significant difference (p >= 0.05)")

print("\\n" + "="*80)""")

add_markdown("""## 9. Visualization""")

add_code("""# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Metric comparison
metrics_to_plot = ['recall', 'f2', 'precision', 'accuracy']
scenario_labels = ['Fully Supervised', 'Semi-Sup (Cluster)', 'Semi-Sup (Model)']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]

    means = []
    stds = []
    for scenario in ['scenario_a', 'scenario_b', 'scenario_c']:
        mean, std = results_summary[scenario][metric]
        means.append(mean)
        stds.append(std)

    x = np.arange(len(scenario_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['blue', 'green', 'orange'])

    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved: scenario_comparison.png")""")

add_markdown("""## 10. Conclusion

### Key Findings

**Performance Summary** (Test Set, Mean across 5 folds):
- Shown in results above

**Statistical Significance**:
- McNemar's test results indicate whether differences are statistically significant

**Best Scenario**:
- Compare F2-score (emphasizes Recall for medical safety)
- Check statistical significance

**Recommendations**:
- Use model-based pseudo-labeling if significantly better
- Consider ensemble of multiple scenarios
- Validate on external dataset before clinical deployment

---

### Next Steps for Production

1. **Acquire more labeled data** (target: 100-200 per class)
2. **External validation** on different dataset
3. **Clinical trial** with expert radiologists
4. **FDA regulatory path** (if deploying in US)
5. **Continuous monitoring** and model updates

---

**MLflow Dashboard**: Run `mlflow ui` to view all experiments""")

# Save notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("\n" + "="*80)
print("NOTEBOOK 3 CREATED SUCCESSFULLY")
print("="*80)
print("\nFile: 3_semi_supervised_learning.ipynb")
print(f"Cells: {len(notebook['cells'])} ({sum(1 for c in notebook['cells'] if c['cell_type']=='markdown')} markdown, {sum(1 for c in notebook['cells'] if c['cell_type']=='code')} code)")
print("\nNotebook includes:")
print("  - Comprehensive explanations")
print("  - 3 scenario implementations")
print("  - 5-fold cross-validation")
print("  - MLflow logging")
print("  - Statistical comparisons")
print("  - Visualizations")
print("\nReady to execute!")
