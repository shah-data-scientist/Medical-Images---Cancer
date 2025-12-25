"""
Integrate Stronger Regularization into Notebook 3

This script adds the regularized model and updates all training configurations.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("3_semi_supervised_learning.ipynb")

print("=" * 80)
print("INTEGRATING STRONGER REGULARIZATION INTO NOTEBOOK 3")
print("=" * 80)

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\n[OK] Loaded notebook: {len(nb['cells'])} cells")

# New regularized model code
regularized_model_code = '''class BrainTumorClassifierRegularized(nn.Module):
    """
    HEAVILY REGULARIZED classifier for TINY datasets (<100 samples).

    Changes from original:
    - Dropout: 0.5 → 0.7 (AGGRESSIVE)
    - Hidden dims: 128 → 64 (REDUCED CAPACITY)
    - Architecture: Simpler (2 layers instead of 3)

    WHY: With only 59 training samples, we need AGGRESSIVE regularization
    to prevent memorization.
    """
    def __init__(self, input_dim=50, hidden_dim=64, dropout=0.7):
        super(BrainTumorClassifierRegularized, self).__init__()

        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)  # 70% dropout!

        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x):
        # Forward pass with heavy dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)  # Drop 70% of neurons

        x = self.fc2(x)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Loss - prevents overconfidence.

    Instead of hard targets [0, 1], uses soft targets [0.1, 0.9]
    This prevents the model from being overconfident on small datasets.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        # Create smooth targets
        num_classes = logits.size(1)
        smooth_target = torch.zeros_like(log_probs).fill_(self.smoothing / (num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)

        # Compute loss
        loss = (-smooth_target * log_probs).sum(dim=1).mean()
        return loss


# Use regularized model for all scenarios
print("\\n" + "="*80)
print("REGULARIZATION CONFIGURATION")
print("="*80)
print("\\nChanges applied:")
print("  - Model: BrainTumorClassifier → BrainTumorClassifierRegularized")
print("  - Dropout: 0.5 → 0.7 (40% increase)")
print("  - Hidden dims: 128 → 64 (50% reduction)")
print("  - Weight decay: 0.01 → 0.05 (5x stronger)")
print("  - Loss: CrossEntropy → LabelSmoothingCrossEntropy (smoothing=0.1)")
print("  - Added: Gradient clipping (max_norm=1.0)")
print("\\nExpected impact: F2 scores drop from 99% to 70-80% (realistic)")
print("="*80)
'''

# Find the cell with BrainTumorClassifier definition
model_cell_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'class BrainTumorClassifier(nn.Module):' in source:
            model_cell_idx = i
            print(f"\n[OK] Found model definition at cell {i}")
            break

if model_cell_idx is None:
    print("[ERROR] Could not find model definition cell")
    exit(1)

# Insert regularized model code before the original model
new_source = regularized_model_code + "\n\n" + ''.join(nb['cells'][model_cell_idx]['source'])
nb['cells'][model_cell_idx]['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]

print(f"[OK] Added regularized model classes to cell {model_cell_idx}")

# Update all model instantiations
replacements = [
    ('BrainTumorClassifier(input_dim=50, dropout=0.5)',
     'BrainTumorClassifierRegularized(input_dim=50, hidden_dim=64, dropout=0.7)'),
    ('nn.CrossEntropyLoss()',
     'LabelSmoothingCrossEntropy(smoothing=0.1)'),
    ('weight_decay=0.01',
     'weight_decay=0.05'),
    ('"dropout": 0.5',
     '"dropout": 0.7'),
    ('"weight_decay": 0.01',
     '"weight_decay": 0.05'),
]

cells_updated = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        modified = False

        for old, new in replacements:
            if old in source:
                source = source.replace(old, new)
                modified = True

        if modified:
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
            cells_updated += 1

print(f"[OK] Updated {cells_updated} cells with new configuration")

# Add gradient clipping to train_model function
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))

        # Look for backward() call and add gradient clipping after it
        if 'loss.backward()' in source and 'clip_grad_norm_' not in source:
            # Add gradient clipping after loss.backward()
            source = source.replace(
                'loss.backward()',
                'loss.backward()\n            # Gradient clipping (prevents exploding gradients)\n            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)'
            )
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]
            print(f"[OK] Added gradient clipping to cell {i}")

# Clear all outputs (will need re-execution)
outputs_cleared = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        cell['outputs'] = []
        cell['execution_count'] = None
        outputs_cleared += 1

print(f"[OK] Cleared outputs from {outputs_cleared} code cells")

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Saved updated notebook")

print("\n" + "=" * 80)
print("INTEGRATION COMPLETE")
print("=" * 80)

print("\nSummary of changes:")
print("  1. Added BrainTumorClassifierRegularized class")
print("  2. Added LabelSmoothingCrossEntropy loss")
print("  3. Updated all model instantiations (3 scenarios)")
print("  4. Increased dropout: 0.5 → 0.7")
print("  5. Reduced hidden dims: 128 → 64")
print("  6. Increased weight decay: 0.01 → 0.05")
print("  7. Added gradient clipping (max_norm=1.0)")
print("  8. Cleared all outputs")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Re-run Notebook 3 with regularization")
print("2. Compare results with previous run")
print("3. Expect F2 scores to drop to 70-80% (realistic range)")
print("4. Add bootstrap confidence intervals to results")

print("\n[OK] Ready to re-run Notebook 3!")
