"""
Curriculum Learning: Train on high-confidence weak labels first,
gradually add lower-confidence labels.

Think of it like teaching a student:
- Start with easy, clear examples
- Gradually introduce harder, ambiguous cases
- Student (model) builds understanding progressively
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def curriculum_learning_scenario(weak_labels_df, unlabeled_pca,
                                  train_idx, val_idx, all_labeled_pca,
                                  all_labeled_labels, device):
    """
    Scenario with curriculum learning on weak labels.

    Strategy:
    1. Sort weak labels by confidence (silhouette score)
    2. Train on top 10% (highest confidence) for 10 epochs
    3. Add next 10% (medium-high confidence) for 10 epochs
    4. Add next 10% (medium confidence) for 10 epochs
    5. Fine-tune on strong labels (as usual)
    """

    # Sort weak labels by confidence
    weak_labels_sorted = weak_labels_df.sort_values('silhouette_score', ascending=False)

    # Define curriculum stages (top 10%, top 20%, top 30%)
    n_total = len(weak_labels_sorted)
    stages = [
        ('Stage 1: Top 10% (Highest Confidence)', int(0.10 * n_total)),
        ('Stage 2: Top 20% (High Confidence)', int(0.20 * n_total)),
        ('Stage 3: Top 30% (Medium Confidence)', int(0.30 * n_total)),
    ]

    # Initialize model
    model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "="*60)
    print("CURRICULUM LEARNING")
    print("="*60)

    # Progressive training through curriculum stages
    for stage_name, n_samples in stages:
        print(f"\n{stage_name}: Training on {n_samples} weak labels")

        # Get weak labels for this stage
        current_weak = weak_labels_sorted.iloc[:n_samples]

        avg_confidence = current_weak['silhouette_score'].mean()
        print(f"  Average confidence: {avg_confidence:.3f}")

        # Create dataset
        weak_features = unlabeled_pca[:n_samples]
        weak_labels = current_weak['weak_label_kmeans'].values

        weak_dataset = FeatureDataset(weak_features, weak_labels)
        weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True)

        # Train for 10 epochs on this stage
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            model.train()
            epoch_loss = 0

            for features, labels in weak_loader:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * features.size(0)

            avg_loss = epoch_loss / len(weak_dataset)

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/10: Loss = {avg_loss:.4f}")

    print("\n  Curriculum pre-training complete!")
    print(f"  Model exposed to {stages[-1][1]} weak labels progressively")

    # Fine-tune on strong labels (same as before)
    print("\n  Fine-tuning on strong labels...")

    train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])
    val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

    print("  Fine-tuning complete!")

    return model


# Key Insight:
# By starting with high-confidence labels, the model learns CORRECT patterns first.
# Then, when exposed to noisier labels, it's less likely to learn the NOISE.
#
# Without curriculum: Model sees noise early, learns wrong patterns
# With curriculum: Model sees clean data first, builds robust foundation
