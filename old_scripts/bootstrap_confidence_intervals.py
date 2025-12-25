"""
Bootstrap Confidence Intervals for Small Test Sets

PURPOSE: With only 30 test samples, a single F2 score is unreliable.
Bootstrap resampling quantifies uncertainty.

EXAMPLE OUTPUT:
  F2 Score: 0.75 ± 0.08 (95% CI: [0.67, 0.83])

This tells us: "We're 95% confident the true F2 is between 0.67 and 0.83"
"""

import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score


def bootstrap_confidence_interval(
    y_true,
    y_pred,
    metric_func,
    n_bootstrap=1000,
    confidence=0.95,
    random_state=42
):
    """
    Calculate bootstrap confidence interval for any metric.

    Args:
        y_true: True labels [n_samples]
        y_pred: Predicted labels [n_samples]
        metric_func: Function that takes (y_true, y_pred) and returns a score
        n_bootstrap: Number of bootstrap samples (default 1000)
        confidence: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        dict with keys:
            - 'mean': Bootstrap mean estimate
            - 'std': Bootstrap standard deviation
            - 'lower': Lower bound of CI
            - 'upper': Upper bound of CI
            - 'original': Metric on original data
    """
    np.random.seed(random_state)

    n_samples = len(y_true)
    bootstrap_scores = []

    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Calculate metric on resampled data
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except:
            # Skip if metric can't be calculated (e.g., all same class)
            continue

    # Calculate statistics
    bootstrap_scores = np.array(bootstrap_scores)

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_scores, alpha * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)

    # Original metric (non-bootstrapped)
    original = metric_func(y_true, y_pred)

    return {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'original': original,
        'n_bootstrap': len(bootstrap_scores)
    }


def f2_score(y_true, y_pred):
    """F2 score (beta=2, emphasizes recall)"""
    return fbeta_score(y_true, y_pred, beta=2, average='binary')


def calculate_all_metrics_with_ci(y_true, y_pred, n_bootstrap=1000, confidence=0.95):
    """
    Calculate all relevant metrics with confidence intervals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        dict with metrics and their confidence intervals
    """
    metrics = {
        'F2 Score': f2_score,
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    }

    results = {}

    for metric_name, metric_func in metrics.items():
        ci = bootstrap_confidence_interval(
            y_true, y_pred, metric_func,
            n_bootstrap=n_bootstrap,
            confidence=confidence
        )
        results[metric_name] = ci

    return results


def print_metrics_with_ci(results, confidence=0.95):
    """
    Pretty print metrics with confidence intervals.

    Args:
        results: Output from calculate_all_metrics_with_ci()
        confidence: Confidence level (for display)
    """
    print("=" * 70)
    print("PERFORMANCE METRICS WITH CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"\nBootstrap samples: {results['F2 Score']['n_bootstrap']}")
    print(f"Confidence level: {confidence*100:.0f}%")
    print(f"\n{'Metric':<15} {'Original':<10} {'Bootstrap Mean':<15} {confidence*100:.0f}% CI")
    print("-" * 70)

    for metric_name, ci in results.items():
        print(f"{metric_name:<15} {ci['original']:.4f}     "
              f"{ci['mean']:.4f} ± {ci['std']:.4f}    "
              f"[{ci['lower']:.4f}, {ci['upper']:.4f}]")

    print("=" * 70)

    # Interpretation
    f2_ci = results['F2 Score']
    ci_width = f2_ci['upper'] - f2_ci['lower']

    print(f"\nINTERPRETATION:")
    print(f"  F2 Score: {f2_ci['original']:.4f} ± {f2_ci['std']:.4f}")
    print(f"  95% CI: [{f2_ci['lower']:.4f}, {f2_ci['upper']:.4f}]")
    print(f"  CI Width: {ci_width:.4f}")
    print(f"\n  → We are 95% confident the true F2 score is between")
    print(f"    {f2_ci['lower']:.4f} and {f2_ci['upper']:.4f}")

    if ci_width > 0.15:
        print(f"\n  ⚠️  Wide confidence interval (>{0.15:.2f}) indicates:")
        print(f"      - High uncertainty due to small test set")
        print(f"      - Need more test data for reliable estimates")
    elif ci_width < 0.05:
        print(f"\n  ✓ Narrow confidence interval (<{0.05:.2f}) indicates:")
        print(f"      - Relatively stable performance estimate")
        print(f"      - Test set size adequate for this metric")
    else:
        print(f"\n  ✓ Moderate confidence interval indicates:")
        print(f"      - Reasonable uncertainty for test set size")
        print(f"      - Performance estimate is fairly reliable")


# Example usage for Notebook 3
"""
HOW TO USE IN NOTEBOOK 3:

After training and evaluating each scenario, add:

```python
from bootstrap_confidence_intervals import (
    calculate_all_metrics_with_ci,
    print_metrics_with_ci
)

# Calculate confidence intervals
results_with_ci = calculate_all_metrics_with_ci(
    y_true=y_test,
    y_pred=y_pred,
    n_bootstrap=1000,
    confidence=0.95
)

# Display results
print_metrics_with_ci(results_with_ci, confidence=0.95)
```

EXPECTED OUTPUT:
======================================================================
PERFORMANCE METRICS WITH CONFIDENCE INTERVALS
======================================================================

Bootstrap samples: 1000
Confidence level: 95%

Metric          Original   Bootstrap Mean  95% CI
----------------------------------------------------------------------
F2 Score        0.7500     0.7485 ± 0.0412    [0.6667, 0.8333]
Accuracy        0.7333     0.7330 ± 0.0425    [0.6500, 0.8167]
Precision       0.7200     0.7215 ± 0.0521    [0.6250, 0.8182]
Recall          0.7800     0.7790 ± 0.0489    [0.6875, 0.8750]
======================================================================

INTERPRETATION:
  F2 Score: 0.7500 ± 0.0412
  95% CI: [0.6667, 0.8333]
  CI Width: 0.1666

  → We are 95% confident the true F2 score is between
    0.6667 and 0.8333

  ⚠️  Wide confidence interval (>0.15) indicates:
      - High uncertainty due to small test set
      - Need more test data for reliable estimates

WHY THIS MATTERS:
- With only 30 test samples, a single score can be misleading
- Bootstrap shows the RANGE of plausible performance values
- Critical for medical AI where reliability matters
- Helps set realistic expectations with stakeholders

WHEN TO USE:
- Always use when test set < 100 samples
- Essential for medical/high-stakes applications
- Include in research papers and reports
- Use to compare scenarios (check if CIs overlap)
"""


if __name__ == "__main__":
    # Example with synthetic data
    print("EXAMPLE: Bootstrap Confidence Intervals on Small Test Set\n")

    # Simulate 30-sample test set
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                       1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                       0, 1, 1, 0, 1, 0, 1, 1, 0, 1])

    # Simulate predictions (75% accurate)
    y_pred = y_true.copy()
    flip_indices = np.random.choice(30, size=8, replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]

    print(f"Test set size: {len(y_true)}")
    print(f"True positives: {(y_true == 1).sum()}")
    print(f"True negatives: {(y_true == 0).sum()}\n")

    # Calculate with confidence intervals
    results = calculate_all_metrics_with_ci(y_true, y_pred, n_bootstrap=1000)
    print_metrics_with_ci(results)
