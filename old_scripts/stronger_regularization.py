"""
Stronger regularization to combat overfitting on small datasets.

Current: dropout=0.5, weight_decay=0.01
New: dropout=0.7, weight_decay=0.05, + additional techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim

class BrainTumorClassifierRegularized(nn.Module):
    """
    Heavily regularized classifier for small datasets.

    Changes from original:
    1. Increased dropout: 0.5 → 0.7
    2. Added dropout after EVERY layer (including final)
    3. Smaller hidden dimensions to reduce capacity
    4. Label smoothing (optional)
    """
    def __init__(self, input_dim=50, hidden_dim=64, dropout=0.7):
        super(BrainTumorClassifierRegularized, self).__init__()

        # Reduced capacity: 128 → 64 for first layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Further reduced: 64 → 32
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        # Final layer with dropout before output
        self.fc3 = nn.Linear(hidden_dim // 2, 2)
        self.dropout3 = nn.Dropout(dropout * 0.5)  # Lighter dropout before output

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.dropout3(x)
        x = self.fc3(x)

        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing: Instead of 0/1 labels, use 0.1/0.9
    Prevents model from being overconfident on small datasets.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        # Convert target to one-hot
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)

        # Apply smoothing
        smooth_target = one_hot * self.confidence + (1 - one_hot) * (self.smoothing / (n_classes - 1))

        # Calculate cross-entropy with smooth targets
        log_probs = torch.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1).mean()

        return loss


def train_with_strong_regularization(train_loader, val_loader, device):
    """
    Training function with all regularization techniques enabled.
    """
    # Model with heavy regularization
    model = BrainTumorClassifierRegularized(
        input_dim=50,
        hidden_dim=64,  # Reduced from 128
        dropout=0.7     # Increased from 0.5
    ).to(device)

    # Label smoothing loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Optimizer with strong weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.05  # Increased from 0.01
    )

    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # Total epochs
        eta_min=1e-6
    )

    # Early stopping with patience
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15  # Increased from 10

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\nTraining with STRONG REGULARIZATION:")
    print(f"  - Dropout: 0.7 (vs. 0.5 baseline)")
    print(f"  - Weight decay: 0.05 (vs. 0.01 baseline)")
    print(f"  - Hidden dim: 64 (vs. 128 baseline)")
    print(f"  - Label smoothing: 0.1")
    print(f"  - Cosine LR schedule")
    print(f"  - Early stopping patience: 15\n")

    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping (additional regularization)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

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
                val_loss += loss.item() * features.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

    return model, history


# Comparison: Expected impact
"""
REGULARIZATION TECHNIQUES & EXPECTED IMPACT:

1. Dropout 0.5 → 0.7:
   - Drops 70% of neurons randomly during training
   - Forces redundant learning across network
   - Impact: -2 to -5% train accuracy, +1 to +3% test accuracy

2. Weight Decay 0.01 → 0.05:
   - Penalizes large weights (L2 regularization)
   - Prefers simpler models
   - Impact: Smoother decision boundaries, better generalization

3. Hidden Dim 128 → 64:
   - Reduces model capacity (fewer parameters)
   - Less ability to memorize training data
   - Impact: -3 to -5% train accuracy, +2 to +4% test accuracy

4. Label Smoothing:
   - Targets become 0.1/0.9 instead of 0/1
   - Prevents overconfidence
   - Impact: Better calibration, +1 to +2% test accuracy

5. Gradient Clipping:
   - Prevents exploding gradients
   - Stabilizes training on small datasets
   - Impact: More stable training, fewer divergence issues

COMBINED EFFECT:
- Training accuracy will DROP (from 100% to ~85-90%)
- Test accuracy should IMPROVE (+3 to +7%)
- Model will be less overconfident
- Better generalization to new data

NOTE: With only 59 training samples, even strong regularization
may not fully prevent overfitting. Consider ensemble methods.
"""
