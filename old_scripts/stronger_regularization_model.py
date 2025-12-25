"""
Stronger Regularization for Notebook 3 - Brain Tumor Classifier

CHANGES from original:
1. Dropout: 0.5 → 0.7 (much more aggressive)
2. Hidden dims: 128 → 64 (reduce capacity)
3. Weight decay: 0.01 → 0.05 (5x stronger)
4. Add label smoothing: 0.1 (prevents overconfidence)
5. Add gradient clipping (prevents exploding gradients)

WHY: 59 training samples is EXTREMELY small. Heavy regularization prevents memorization.
"""

import torch
import torch.nn as nn

class BrainTumorClassifierRegularized(nn.Module):
    """
    Heavily regularized classifier for TINY datasets (<100 samples).

    Architecture:
    - Input: 50D PCA features
    - Hidden: 64 units (reduced from 128)
    - Output: 2 classes (cancer/normal)
    - Dropout: 0.7 (very aggressive)
    """
    def __init__(self, input_dim=50, hidden_dim=64, dropout=0.7):
        super().__init__()

        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.dropout1 = nn.Dropout(dropout)     # 70% dropout!

        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # Forward pass with heavy dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)  # Drop 70% of neurons during training
        x = self.fc2(x)
        return x  # Return logits (not probabilities)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Loss - prevents overconfidence on small datasets.

    Instead of hard targets [0, 1] or [1, 0]:
    Uses soft targets [0.1, 0.9] or [0.9, 0.1]

    WHY: With only 59 training samples, model can easily memorize.
    Label smoothing forces model to be less confident → better generalization.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        """
        Args:
            logits: Model output (before softmax) [batch_size, num_classes]
            target: True labels [batch_size]
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        # Create smooth targets
        num_classes = logits.size(1)
        smooth_target = torch.zeros_like(log_probs).fill_(self.smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)

        # Compute loss
        loss = (-smooth_target * log_probs).sum(dim=1).mean()
        return loss


# Training configuration with stronger regularization
TRAINING_CONFIG_REGULARIZED = {
    # Model architecture
    'input_dim': 50,
    'hidden_dim': 64,      # Reduced from 128 (less capacity)
    'dropout': 0.7,        # Increased from 0.5 (more dropout)

    # Optimizer
    'learning_rate': 0.001,
    'weight_decay': 0.05,  # Increased from 0.01 (5x stronger L2 penalty)

    # Loss function
    'label_smoothing': 0.1,  # NEW: Prevents overconfidence

    # Training
    'batch_size': 16,      # Keep small for tiny dataset
    'max_epochs': 100,

    # Gradient clipping (NEW)
    'max_grad_norm': 1.0,  # Clip gradients to prevent exploding

    # Early stopping
    'patience': 15,        # Stop if no improvement for 15 epochs
}


def train_with_stronger_regularization(model, train_loader, val_loader, config):
    """
    Training loop with stronger regularization techniques.

    NEW additions:
    1. Label smoothing loss
    2. Gradient clipping
    3. Higher weight decay
    """
    import torch.optim as optim
    from tqdm import tqdm

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])

    # Optimizer with stronger weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # L2 regularization
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)

            # Backward pass
            loss.backward()

            # GRADIENT CLIPPING (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['max_grad_norm']
            )

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model


# Example usage comment for Notebook 3
"""
HOW TO USE IN NOTEBOOK 3:

# Replace the original BrainTumorClassifier with this version:
from stronger_regularization_model import (
    BrainTumorClassifierRegularized,
    LabelSmoothingCrossEntropy,
    TRAINING_CONFIG_REGULARIZED,
    train_with_stronger_regularization
)

# Initialize model with stronger regularization
model = BrainTumorClassifierRegularized(
    input_dim=50,
    hidden_dim=64,   # Reduced capacity
    dropout=0.7      # Heavy dropout
)

# Train with label smoothing + gradient clipping
model = train_with_stronger_regularization(
    model, train_loader, val_loader, TRAINING_CONFIG_REGULARIZED
)

EXPECTED IMPACT:
- Lower training accuracy (prevents memorization)
- Better generalization to test set
- More realistic performance estimates
- Reduced overfitting (100% train → ~85% train)
"""
