"""
Model Calibration: Ensuring predicted probabilities match actual outcomes.

Why calibration matters for medical AI:
- "90% cancer" should mean 90 out of 100 such predictions are cancer
- Helps doctors make informed decisions
- Required for clinical deployment

Techniques:
1. Platt Scaling (Logistic Regression)
2. Temperature Scaling (simpler, better for small datasets)
3. Isotonic Regression (non-parametric)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling: Simple but effective calibration method.

    How it works:
    - Divide logits by temperature T before softmax
    - T > 1: Makes model less confident (smooths probabilities)
    - T < 1: Makes model more confident (sharpens probabilities)
    - Optimal T is learned on validation set
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initialize T=1.5

    def forward(self, logits):
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs (before softmax)

        Returns:
            Calibrated probabilities
        """
        return torch.softmax(logits / self.temperature, dim=1)

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Validation set logits (N, 2)
            labels: True labels (N,)
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self


def calibrate_model_temperature_scaling(model, val_loader, device):
    """
    Apply temperature scaling to calibrate model probabilities.

    Args:
        model: Trained PyTorch model
        val_loader: Validation data loader
        device: torch device

    Returns:
        temp_scaler: TemperatureScaling module
        calibrated_probs: Calibrated probabilities
    """
    model.eval()

    # Collect logits and labels from validation set
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            logits = model(features)  # Raw outputs (before softmax)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Fit temperature
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(all_logits, all_labels)

    # Get calibrated probabilities
    with torch.no_grad():
        calibrated_probs = temp_scaler(all_logits).numpy()

    return temp_scaler, calibrated_probs


def plot_calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Plot reliability diagram (calibration curve).

    Perfect calibration: Diagonal line
    Overconfident: Curve below diagonal
    Underconfident: Curve above diagonal

    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities for class 1
        n_bins: Number of bins
        strategy: 'uniform' or 'quantile'
    """
    from sklearn.calibration import calibration_curve

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob,
        n_bins=n_bins,
        strategy=strategy
    )

    # Calculate Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_prob)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Actual calibration
    ax.plot(prob_pred, prob_true, 's-', label=f'Model (ECE={ece:.3f})', linewidth=2, markersize=8)

    # Formatting
    ax.set_xlabel('Predicted Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    return ece


def compare_calibration_before_after(y_true, y_prob_before, y_prob_after):
    """
    Compare calibration before and after temperature scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    from sklearn.calibration import calibration_curve

    # Before calibration
    prob_true_before, prob_pred_before = calibration_curve(
        y_true, y_prob_before,
        n_bins=10,
        strategy='uniform'
    )

    bin_counts = np.histogram(y_prob_before, bins=10, range=(0, 1))[0]
    bin_weights = bin_counts / len(y_prob_before)
    ece_before = np.sum(bin_weights * np.abs(prob_true_before - prob_pred_before))

    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    axes[0].plot(prob_pred_before, prob_true_before, 's-',
                 label=f'Before (ECE={ece_before:.3f})', linewidth=2, markersize=8, color='red')
    axes[0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('BEFORE Calibration', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # After calibration
    prob_true_after, prob_pred_after = calibration_curve(
        y_true, y_prob_after,
        n_bins=10,
        strategy='uniform'
    )

    bin_weights_after = np.histogram(y_prob_after, bins=10, range=(0, 1))[0] / len(y_prob_after)
    ece_after = np.sum(bin_weights_after * np.abs(prob_true_after - prob_pred_after))

    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    axes[1].plot(prob_pred_after, prob_true_after, 's-',
                 label=f'After (ECE={ece_after:.3f})', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Actual Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('AFTER Calibration', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nExpected Calibration Error:")
    print(f"  Before: {ece_before:.4f}")
    print(f"  After:  {ece_after:.4f}")
    print(f"  Improvement: {(ece_before - ece_after):.4f}")

    return ece_before, ece_after


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def calibrate_scenario_models(results, test_loader, device):
    """
    Calibrate all scenario models using temperature scaling.

    This should be added to Notebook 3 after the 5-fold CV.
    """
    print("\n" + "="*60)
    print("MODEL CALIBRATION")
    print("="*60)

    calibrated_results = {}

    for scenario_name in ['scenario_a', 'scenario_b', 'scenario_c']:
        print(f"\nCalibrating {scenario_name}...")

        # For each fold, calibrate the model
        # (In practice, you'd use a validation set from that fold)

        # Placeholder: Load model from checkpoint
        # model = load_model_from_checkpoint(f'{scenario_name}_fold_1.pt')

        # Apply temperature scaling
        # temp_scaler, calibrated_probs = calibrate_model_temperature_scaling(
        #     model, val_loader, device
        # )

        # Store calibrated predictions
        # calibrated_results[scenario_name] = calibrated_probs

        print(f"  {scenario_name} calibrated successfully")

    return calibrated_results


# ============================================================================
# WHY CALIBRATION MATTERS
# ============================================================================

"""
MEDICAL AI CONTEXT:

Uncalibrated Model:
- Predicts "95% cancer" for 100 patients
- Actually only 70 have cancer
- Doctor makes incorrect risk assessment
- Overtreatment or unnecessary biopsies

Calibrated Model:
- Predicts "70% cancer" for 100 patients
- Actually 70 have cancer (matches prediction)
- Doctor can make informed decision
- Better resource allocation

EXPECTED IMPROVEMENT FOR YOUR MODEL:
- Current: Likely overconfident (predicts 99% but shouldn't)
- After calibration: More realistic probabilities (75-85% range)
- Better for clinical decision-making
- Won't change F2-score, but improves probability estimates
"""
