"""
Advanced Validation Analysis - Week 1 Strategies
Implements validation techniques from alternative_validation_plan.md

This script performs:
1. Feature importance via permutation
2. t-SNE visualization of feature space
3. Ensemble modeling (combining 5-fold models)
4. Noise injection robustness testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# 1. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, X_test, y_test, n_repeats=30, random_state=42):
    """
    Perform permutation-based feature importance analysis.

    Args:
        model: Trained model (must have predict method)
        X_test: Test features (n_samples, n_features)
        y_test: Test labels
        n_repeats: Number of permutations
        random_state: Random seed

    Returns:
        dict: Feature importance results
    """
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS (Permutation-Based)")
    print("=" * 80)

    np.random.seed(random_state)

    # Get baseline score
    baseline_preds = model.predict(X_test)
    baseline_score = fbeta_score(y_test, baseline_preds, beta=2)

    print(f"\nBaseline F2 Score: {baseline_score:.4f}")
    print(f"Running {n_repeats} permutations for each of {X_test.shape[1]} features...")

    # Compute permutation importance manually
    n_features = X_test.shape[1]
    importances = np.zeros((n_features, n_repeats))

    for feature_idx in range(n_features):
        for repeat in range(n_repeats):
            # Copy data and permute this feature
            X_permuted = X_test.copy()
            X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

            # Get score with permuted feature
            permuted_preds = model.predict(X_permuted)
            permuted_score = fbeta_score(y_test, permuted_preds, beta=2)

            # Importance = drop in performance
            importances[feature_idx, repeat] = baseline_score - permuted_score

        if (feature_idx + 1) % 10 == 0:
            print(f"  Processed {feature_idx + 1}/{n_features} features...")

    # Get importance scores
    importances_mean = importances.mean(axis=1)
    importances_std = importances.std(axis=1)

    # Identify critical components (importance > 0.01)
    critical_threshold = 0.01
    critical_components = np.where(importances_mean > critical_threshold)[0]

    print(f"\nTotal PCA components: {len(importances_mean)}")
    print(f"Critical components (importance > {critical_threshold}): {len(critical_components)}")
    print(f"Critical component indices: {critical_components.tolist()}")

    # Top 10 most important components
    top_10_indices = np.argsort(importances_mean)[-10:][::-1]
    print(f"\nTop 10 most important components:")
    for i, idx in enumerate(top_10_indices, 1):
        print(f"  {i}. Component {idx}: {importances_mean[idx]:.4f} ± {importances_std[idx]:.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: All components
    axes[0].bar(range(len(importances_mean)), importances_mean,
                yerr=importances_std, alpha=0.7, capsize=2)
    axes[0].axhline(y=critical_threshold, color='r', linestyle='--',
                    label=f'Threshold ({critical_threshold})')
    axes[0].set_xlabel('PCA Component Index')
    axes[0].set_ylabel('Permutation Importance')
    axes[0].set_title('Feature Importance: All Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Top 10 components
    axes[1].barh(range(10), importances_mean[top_10_indices],
                 xerr=importances_std[top_10_indices], alpha=0.7, capsize=3)
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([f'PC{idx}' for idx in top_10_indices])
    axes[1].set_xlabel('Permutation Importance')
    axes[1].set_title('Top 10 Most Important Components')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'importances_mean': importances_mean,
        'importances_std': importances_std,
        'critical_components': critical_components,
        'top_10_components': top_10_indices,
        'n_critical': len(critical_components)
    }


# ============================================================================
# 2. t-SNE VISUALIZATION
# ============================================================================

def visualize_feature_space_tsne(X_train, y_train, X_test, y_test,
                                  random_state=42, perplexity=30):
    """
    Visualize feature space using t-SNE dimensionality reduction.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed
        perplexity: t-SNE perplexity parameter

    Returns:
        dict: t-SNE results
    """
    print("\n" + "=" * 80)
    print("t-SNE VISUALIZATION OF FEATURE SPACE")
    print("=" * 80)

    # Combine train and test for visualization
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    split_labels = ['Train'] * len(X_train) + ['Test'] * len(X_test)

    print(f"\nComputing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_combined)

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Plot 1: Color by true labels
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

    # Plot 2: Color by train/test split
    for i, split in enumerate(['Train', 'Test']):
        mask = np.array(split_labels) == split
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       label=split, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].set_title('t-SNE: Colored by Train/Test Split')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Combined view (label + split)
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
    plt.show()

    # Calculate separation metrics
    train_normal = X_tsne[(np.array(split_labels) == 'Train') & (y_combined == 0)]
    train_cancer = X_tsne[(np.array(split_labels) == 'Train') & (y_combined == 1)]

    # Mean distance between class centroids
    centroid_normal = train_normal.mean(axis=0)
    centroid_cancer = train_cancer.mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_normal - centroid_cancer)

    print(f"\nClass Separation Metrics:")
    print(f"  Distance between class centroids: {centroid_distance:.2f}")
    print(f"  Normal cluster std: {train_normal.std(axis=0).mean():.2f}")
    print(f"  Cancer cluster std: {train_cancer.std(axis=0).mean():.2f}")

    return {
        'X_tsne': X_tsne,
        'centroid_distance': centroid_distance,
        'centroid_normal': centroid_normal,
        'centroid_cancer': centroid_cancer
    }


# ============================================================================
# 3. ENSEMBLE MODELING
# ============================================================================

def create_ensemble_predictions(fold_models, X_test, y_test):
    """
    Create ensemble predictions by averaging all 5 fold models.

    Args:
        fold_models: List of 5 trained models (one per fold)
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Ensemble results
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE MODELING (5-Fold Averaging)")
    print("=" * 80)

    # Collect predictions from all models
    all_probs = []
    individual_f2_scores = []

    print("\nIndividual fold performance on test set:")
    for i, model in enumerate(fold_models, 1):
        # Get probabilities
        probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        all_probs.append(probs)

        # Individual predictions
        preds = (probs > 0.5).astype(int)
        f2 = fbeta_score(y_test, preds, beta=2)
        individual_f2_scores.append(f2)
        print(f"  Fold {i}: F2 = {f2:.4f}")

    # Average probabilities across all folds
    all_probs = np.array(all_probs)
    ensemble_probs = all_probs.mean(axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # Calculate ensemble metrics
    ensemble_f2 = fbeta_score(y_test, ensemble_preds, beta=2)
    ensemble_precision = precision_score(y_test, ensemble_preds, zero_division=0)
    ensemble_recall = recall_score(y_test, ensemble_preds)
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

    print(f"\nEnsemble Results:")
    print(f"  F2 Score:    {ensemble_f2:.4f}")
    print(f"  Precision:   {ensemble_precision:.4f}")
    print(f"  Recall:      {ensemble_recall:.4f}")
    print(f"  Accuracy:    {ensemble_accuracy:.4f}")

    # Improvement over average individual model
    avg_individual_f2 = np.mean(individual_f2_scores)
    improvement = ensemble_f2 - avg_individual_f2
    print(f"\nImprovement over average individual model: {improvement:+.4f} ({improvement/avg_individual_f2*100:+.2f}%)")

    # Prediction confidence analysis
    high_confidence = ensemble_probs.max() > 0.9 if ensemble_probs.max() <= 1.0 else (ensemble_probs > 0.9).any()
    low_confidence = ensemble_probs.min() < 0.7 if ensemble_probs.min() >= 0.0 else (ensemble_probs < 0.7).any()

    # Calculate confidence metrics differently
    confident_mask = (ensemble_probs > 0.9) | (ensemble_probs < 0.1)
    uncertain_mask = (ensemble_probs >= 0.3) & (ensemble_probs <= 0.7)

    print(f"\nPrediction Confidence:")
    print(f"  High confidence predictions (>0.9 or <0.1): {confident_mask.sum()}/{len(y_test)}")
    print(f"  Uncertain predictions (0.3-0.7): {uncertain_mask.sum()}/{len(y_test)}")

    if confident_mask.sum() > 0:
        conf_accuracy = accuracy_score(y_test[confident_mask], ensemble_preds[confident_mask])
        print(f"  High confidence accuracy: {conf_accuracy:.2%}")

    if uncertain_mask.sum() > 0:
        uncertain_accuracy = accuracy_score(y_test[uncertain_mask], ensemble_preds[uncertain_mask])
        print(f"  Uncertain predictions accuracy: {uncertain_accuracy:.2%}")

    # Visualize probability distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Ensemble probability distribution
    axes[0].hist(ensemble_probs, bins=20, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold')
    axes[0].set_xlabel('Ensemble Probability (Class 1)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Ensemble Probabilities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Variance across folds
    prob_variance = all_probs.var(axis=0)
    axes[1].scatter(ensemble_probs, prob_variance, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Ensemble Probability (Class 1)')
    axes[1].set_ylabel('Variance Across Folds')
    axes[1].set_title('Prediction Uncertainty (Fold Variance)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ensemble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'ensemble_probs': ensemble_probs,
        'ensemble_preds': ensemble_preds,
        'f2': ensemble_f2,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'accuracy': ensemble_accuracy,
        'individual_f2_scores': individual_f2_scores,
        'improvement': improvement
    }


# ============================================================================
# 4. NOISE INJECTION ROBUSTNESS TEST
# ============================================================================

def test_noise_robustness(model, X_test, y_test, noise_levels=None):
    """
    Test model robustness by adding controlled noise to features.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        noise_levels: List of noise standard deviations to test

    Returns:
        dict: Robustness test results
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("\n" + "=" * 80)
    print("NOISE INJECTION ROBUSTNESS TEST")
    print("=" * 80)

    results = {
        'noise_levels': [],
        'f2_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'accuracy_scores': []
    }

    print(f"\nTesting with noise levels: {noise_levels}")
    print(f"{'Noise Std':<12} {'F2':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10}")
    print("-" * 60)

    for noise_std in noise_levels:
        # Add Gaussian noise to features
        if noise_std > 0:
            noise = np.random.randn(*X_test.shape) * noise_std
            X_noisy = X_test + noise
        else:
            X_noisy = X_test

        # Evaluate on noisy data
        y_pred = model.predict(X_noisy)

        f2 = fbeta_score(y_test, y_pred, beta=2)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        results['noise_levels'].append(noise_std)
        results['f2_scores'].append(f2)
        results['precision_scores'].append(precision)
        results['recall_scores'].append(recall)
        results['accuracy_scores'].append(accuracy)

        print(f"{noise_std:<12.2f} {f2:<10.4f} {precision:<12.4f} {recall:<10.4f} {accuracy:<10.4f}")

    # Calculate robustness metrics
    baseline_f2 = results['f2_scores'][0]
    f2_at_10pct_noise = results['f2_scores'][noise_levels.index(0.10)]
    robustness_score = f2_at_10pct_noise / baseline_f2

    print(f"\nRobustness Metrics:")
    print(f"  Baseline F2 (no noise):     {baseline_f2:.4f}")
    print(f"  F2 at 10% noise:            {f2_at_10pct_noise:.4f}")
    print(f"  Robustness score (ratio):   {robustness_score:.4f}")
    print(f"  Performance drop at 10%:    {(1-robustness_score)*100:.2f}%")

    # Target: F2 should stay >80% with 10% noise
    target_f2 = 0.80
    if f2_at_10pct_noise > target_f2:
        print(f"  Status: PASS (F2 > {target_f2} with 10% noise)")
    else:
        print(f"  Status: FAIL (F2 < {target_f2} with 10% noise)")

    # Visualize robustness curves
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
        ax.set_xlabel('Noise Level (Standard Deviation)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs. Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('noise_robustness_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_all_validation_analyses(model, fold_models, X_train, y_train, X_test, y_test):
    """
    Run all validation analyses and save results.

    Args:
        model: Single trained model (e.g., best fold)
        fold_models: List of all 5 fold models
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: All analysis results
    """
    results = {}

    # 1. Feature Importance
    print("\n\nSTARTING VALIDATION ANALYSIS...")
    print("=" * 80)
    results['feature_importance'] = analyze_feature_importance(
        model, X_test, y_test, n_repeats=30
    )

    # 2. t-SNE Visualization
    results['tsne'] = visualize_feature_space_tsne(
        X_train, y_train, X_test, y_test
    )

    # 3. Ensemble Modeling
    results['ensemble'] = create_ensemble_predictions(
        fold_models, X_test, y_test
    )

    # 4. Noise Robustness
    results['robustness'] = test_noise_robustness(
        model, X_test, y_test
    )

    # Save results to JSON
    results_serializable = {
        'feature_importance': {
            'n_critical_components': int(results['feature_importance']['n_critical']),
            'critical_component_indices': results['feature_importance']['critical_components'].tolist(),
            'top_10_component_indices': results['feature_importance']['top_10_components'].tolist()
        },
        'tsne': {
            'centroid_distance': float(results['tsne']['centroid_distance'])
        },
        'ensemble': {
            'f2': float(results['ensemble']['f2']),
            'precision': float(results['ensemble']['precision']),
            'recall': float(results['ensemble']['recall']),
            'accuracy': float(results['ensemble']['accuracy']),
            'improvement_over_individual': float(results['ensemble']['improvement'])
        },
        'robustness': {
            'noise_levels': results['robustness']['noise_levels'],
            'f2_scores': results['robustness']['f2_scores'],
            'f2_at_10pct_noise': float(results['robustness']['f2_scores'][2]),  # index 2 = 0.10 noise
            'robustness_score': float(results['robustness']['f2_scores'][2] / results['robustness']['f2_scores'][0])
        }
    }

    with open('validation_analysis_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - validation_analysis_results.json")
    print("  - feature_importance_analysis.png")
    print("  - tsne_visualization.png")
    print("  - ensemble_analysis.png")
    print("  - noise_robustness_test.png")

    return results


if __name__ == "__main__":
    print("Advanced Validation Analysis Script")
    print("This script requires:")
    print("  - model: Trained model (best fold)")
    print("  - fold_models: List of 5 trained models")
    print("  - X_train, y_train: Training data")
    print("  - X_test, y_test: Test data")
    print("\nImport this module and call run_all_validation_analyses()")
