"""
Standalone script to run all validation analyses.
Executes: Feature Importance, t-SNE, and Noise Robustness tests.
Integrates with MLflow for experiment tracking.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import mlflow
import sys
import os

# Ensure we can import the sibling module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import advanced_validation_analysis as ava

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("VALIDATION ANALYSIS - STANDALONE EXECUTION")
print("=" * 80)

# Load features
print("\nLoading features...")
try:
    features_pca_50 = np.load('features/features_pca_50.npy')
    weak_labels_df = pd.read_csv('features/weak_labels.csv')
except FileNotFoundError:
    print("Error: Could not find feature files. Please run from project root.")
    sys.exit(1)

# Get labeled samples only
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
# STEP 2-4: RUN ADVANCED ANALYSES WITH MLFLOW
# ============================================================================
print("\n" + "=" * 80)
print("STARTING MLFLOW VALIDATION RUN")
print("=" * 80)

# Set experiment
mlflow.set_experiment("Validation_Analysis_Standalone")

with mlflow.start_run(run_name="Full_Validation_Suite") as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    
    # Log Hyperparameters
    mlflow.log_params({
        "model_type": "RegularizedMLP",
        "hidden_size": 64,
        "dropout_rate": 0.70,
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "epochs": 50
    })

    # 1. Feature Importance
    print("\nRunning Feature Importance Analysis...")
    ava.analyze_feature_importance(wrapped_model, test_pca, test_labels)

    # 2. t-SNE Visualization
    print("\nRunning t-SNE Visualization...")
    ava.visualize_feature_space_tsne(training_pool_pca, training_pool_labels, test_pca, test_labels)

    # 3. Noise Robustness
    print("\nRunning Noise Robustness Test...")
    ava.test_noise_robustness(wrapped_model, test_pca, test_labels)

print("\n" + "=" * 80)
print("VALIDATION ANALYSIS COMPLETE")
print("=" * 80)
print("Results have been logged to MLflow.")
print("Run 'mlflow ui' to view the results.")