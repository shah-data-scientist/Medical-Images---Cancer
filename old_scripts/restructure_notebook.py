"""
Restructure Notebook 3 to remove confusion between Scenarios and Approaches.

This script:
1. Removes old "Approach 1 & 2" cells (cells 23+)
2. Enhances Cell 22 with comprehensive results aggregation
3. Adds statistical comparison and visualizations
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("3_semi_supervised_learning.ipynb")

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Original notebook: {len(nb['cells'])} cells")

# Keep only cells 0-22 (remove cells 23+)
# Cells 0-22 include: setup, data loading, model definitions, 3 scenarios, CV, budget analysis
nb['cells'] = nb['cells'][:23]

print(f"After removing old Approach cells: {len(nb['cells'])} cells")

# ============================================================================
# Enhanced Cell 22: Comprehensive Results Aggregation & Comparison
# ============================================================================

enhanced_cell_22_source = [
    "# ============================================================================\n",
    "# RESULTS AGGREGATION & COMPREHENSIVE COMPARISON\n",
    "# ============================================================================\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from scipy import stats\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "print(\"=\"*80)\n",
    "print(\"AGGREGATING RESULTS FROM 5-FOLD CROSS-VALIDATION\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "# Extract metrics from results\n",
    "def aggregate_scenario_results(results_list, scenario_name):\n",
    "    \"\"\"Aggregate metrics across all folds for a scenario.\"\"\"\n",
    "    metrics_by_fold = []\n",
    "    \n",
    "    for fold_metrics, _, _ in results_list:\n",
    "        metrics_by_fold.append({\n",
    "            'f2': fold_metrics['f2'],\n",
    "            'recall': fold_metrics['recall'],\n",
    "            'precision': fold_metrics['precision'],\n",
    "            'f1': fold_metrics['f1'],\n",
    "            'accuracy': fold_metrics['accuracy']\n",
    "        })\n",
    "    \n",
    "    df = pd.DataFrame(metrics_by_fold)\n",
    "    \n",
    "    # Calculate mean and std\n",
    "    summary = {\n",
    "        'Scenario': scenario_name,\n",
    "        'F2 Mean': df['f2'].mean(),\n",
    "        'F2 Std': df['f2'].std(),\n",
    "        'Recall Mean': df['recall'].mean(),\n",
    "        'Recall Std': df['recall'].std(),\n",
    "        'Precision Mean': df['precision'].mean(),\n",
    "        'Precision Std': df['precision'].std(),\n",
    "        'F1 Mean': df['f1'].mean(),\n",
    "        'F1 Std': df['f1'].std(),\n",
    "        'Accuracy Mean': df['accuracy'].mean(),\n",
    "        'Accuracy Std': df['accuracy'].std()\n",
    "    }\n",
    "    \n",
    "    return summary, df\n",
    "\n",
    "# Aggregate results for all scenarios\n",
    "summary_a, df_a = aggregate_scenario_results(results['scenario_a'], 'A: Fully Supervised')\n",
    "summary_b, df_b = aggregate_scenario_results(results['scenario_b'], 'B: Clustering Semi-Sup')\n",
    "summary_c, df_c = aggregate_scenario_results(results['scenario_c'], 'C: Model Semi-Sup')\n",
    "\n",
    "# Create comparison DataFrame\n",
    "comparison_df = pd.DataFrame([summary_a, summary_b, summary_c])\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"SCENARIO COMPARISON (Mean ± Std across 5 folds)\")\n",
    "print(\"=\"*80)\n",
    "print(comparison_df.to_string(index=False))\n",
    "\n",
    "# ============================================================================\n",
    "# Statistical Significance Testing (Paired t-tests)\n",
    "# ============================================================================\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"STATISTICAL SIGNIFICANCE TESTS (Paired t-tests on F2 scores)\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "# Extract F2 scores for each scenario\n",
    "f2_a = df_a['f2'].values\n",
    "f2_b = df_b['f2'].values\n",
    "f2_c = df_c['f2'].values\n",
    "\n",
    "# Paired t-tests\n",
    "t_stat_ab, p_value_ab = stats.ttest_rel(f2_a, f2_b)\n",
    "t_stat_ac, p_value_ac = stats.ttest_rel(f2_a, f2_c)\n",
    "t_stat_bc, p_value_bc = stats.ttest_rel(f2_b, f2_c)\n",
    "\n",
    "print(f\"\\nScenario A vs B:\")\n",
    "print(f\"  t-statistic: {t_stat_ab:.4f}\")\n",
    "print(f\"  p-value: {p_value_ab:.4f}\")\n",
    "print(f\"  Significant (α=0.05): {'YES' if p_value_ab < 0.05 else 'NO'}\")\n",
    "\n",
    "print(f\"\\nScenario A vs C:\")\n",
    "print(f\"  t-statistic: {t_stat_ac:.4f}\")\n",
    "print(f\"  p-value: {p_value_ac:.4f}\")\n",
    "print(f\"  Significant (α=0.05): {'YES' if p_value_ac < 0.05 else 'NO'}\")\n",
    "\n",
    "print(f\"\\nScenario B vs C:\")\n",
    "print(f\"  t-statistic: {t_stat_bc:.4f}\")\n",
    "print(f\"  p-value: {p_value_bc:.4f}\")\n",
    "print(f\"  Significant (α=0.05): {'YES' if p_value_bc < 0.05 else 'NO'}\")\n",
    "\n",
    "# ============================================================================\n",
    "# Visualization 1: Box Plot Comparison\n",
    "# ============================================================================\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
    "\n",
    "# F2 Score comparison\n",
    "ax1 = axes[0]\n",
    "f2_data = [f2_a, f2_b, f2_c]\n",
    "bp1 = ax1.boxplot(f2_data, labels=['Scenario A', 'Scenario B', 'Scenario C'], patch_artist=True)\n",
    "for patch, color in zip(bp1['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):\n",
    "    patch.set_facecolor(color)\n",
    "    patch.set_alpha(0.6)\n",
    "ax1.set_ylabel('F2 Score', fontsize=12, fontweight='bold')\n",
    "ax1.set_title('F2 Score Distribution', fontsize=13, fontweight='bold')\n",
    "ax1.grid(axis='y', alpha=0.3)\n",
    "ax1.set_ylim(0, 1.1)\n",
    "\n",
    "# Recall comparison\n",
    "ax2 = axes[1]\n",
    "recall_data = [df_a['recall'].values, df_b['recall'].values, df_c['recall'].values]\n",
    "bp2 = ax2.boxplot(recall_data, labels=['Scenario A', 'Scenario B', 'Scenario C'], patch_artist=True)\n",
    "for patch, color in zip(bp2['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):\n",
    "    patch.set_facecolor(color)\n",
    "    patch.set_alpha(0.6)\n",
    "ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')\n",
    "ax2.set_title('Recall Distribution', fontsize=13, fontweight='bold')\n",
    "ax2.grid(axis='y', alpha=0.3)\n",
    "ax2.set_ylim(0, 1.1)\n",
    "\n",
    "# Precision comparison\n",
    "ax3 = axes[2]\n",
    "precision_data = [df_a['precision'].values, df_b['precision'].values, df_c['precision'].values]\n",
    "bp3 = ax3.boxplot(precision_data, labels=['Scenario A', 'Scenario B', 'Scenario C'], patch_artist=True)\n",
    "for patch, color in zip(bp3['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):\n",
    "    patch.set_facecolor(color)\n",
    "    patch.set_alpha(0.6)\n",
    "ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')\n",
    "ax3.set_title('Precision Distribution', fontsize=13, fontweight='bold')\n",
    "ax3.grid(axis='y', alpha=0.3)\n",
    "ax3.set_ylim(0, 1.1)\n",
    "\n",
    "plt.suptitle('Performance Metrics Comparison Across 5 Folds', fontsize=16, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# ============================================================================\n",
    "# Visualization 2: Mean ± Std Bar Chart\n",
    "# ============================================================================\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 6))\n",
    "\n",
    "metrics = ['F2', 'Recall', 'Precision', 'F1', 'Accuracy']\n",
    "x = np.arange(len(metrics))\n",
    "width = 0.25\n",
    "\n",
    "means_a = [summary_a[f'{m} Mean'] for m in metrics]\n",
    "stds_a = [summary_a[f'{m} Std'] for m in metrics]\n",
    "means_b = [summary_b[f'{m} Mean'] for m in metrics]\n",
    "stds_b = [summary_b[f'{m} Std'] for m in metrics]\n",
    "means_c = [summary_c[f'{m} Mean'] for m in metrics]\n",
    "stds_c = [summary_c[f'{m} Std'] for m in metrics]\n",
    "\n",
    "ax.bar(x - width, means_a, width, yerr=stds_a, label='Scenario A', color='#3498db', alpha=0.7, capsize=5)\n",
    "ax.bar(x, means_b, width, yerr=stds_b, label='Scenario B', color='#e74c3c', alpha=0.7, capsize=5)\n",
    "ax.bar(x + width, means_c, width, yerr=stds_c, label='Scenario C', color='#2ecc71', alpha=0.7, capsize=5)\n",
    "\n",
    "ax.set_xlabel('Metric', fontsize=12, fontweight='bold')\n",
    "ax.set_ylabel('Score (Mean ± Std)', fontsize=12, fontweight='bold')\n",
    "ax.set_title('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(metrics)\n",
    "ax.legend(fontsize=11)\n",
    "ax.grid(axis='y', alpha=0.3)\n",
    "ax.set_ylim(0, 1.1)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# ============================================================================\n",
    "# Save Results\n",
    "# ============================================================================\n",
    "\n",
    "# Save comparison table\n",
    "comparison_df.to_csv('scenario_comparison.csv', index=False)\n",
    "print(f\"\\n✓ Scenario comparison saved to: scenario_comparison.csv\")\n",
    "\n",
    "# Save detailed results\n",
    "detailed_results = {\n",
    "    'scenario_a': df_a.to_dict('records'),\n",
    "    'scenario_b': df_b.to_dict('records'),\n",
    "    'scenario_c': df_c.to_dict('records'),\n",
    "    'statistical_tests': {\n",
    "        'a_vs_b': {'t_stat': float(t_stat_ab), 'p_value': float(p_value_ab)},\n",
    "        'a_vs_c': {'t_stat': float(t_stat_ac), 'p_value': float(p_value_ac)},\n",
    "        'b_vs_c': {'t_stat': float(t_stat_bc), 'p_value': float(p_value_bc)}\n",
    "    }\n",
    "}\n",
    "\n",
    "import json\n",
    "with open('detailed_cv_results.json', 'w') as f:\n",
    "    json.dump(detailed_results, f, indent=2)\n",
    "print(f\"✓ Detailed CV results saved to: detailed_cv_results.json\")\n",
    "\n",
    "# ============================================================================\n",
    "# CONCLUSIONS\n",
    "# ============================================================================\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n",
    "print(\"CONCLUSIONS\")\n",
    "print(\"=\"*80)\n",
    "\n",
    "# Find best scenario\n",
    "best_f2_mean = max(summary_a['F2 Mean'], summary_b['F2 Mean'], summary_c['F2 Mean'])\n",
    "if summary_a['F2 Mean'] == best_f2_mean:\n",
    "    best_scenario = \"Scenario A (Fully Supervised)\"\n",
    "elif summary_b['F2 Mean'] == best_f2_mean:\n",
    "    best_scenario = \"Scenario B (Clustering Semi-Supervised)\"\n",
    "else:\n",
    "    best_scenario = \"Scenario C (Model-based Semi-Supervised)\"\n",
    "\n",
    "print(f\"\\nBest Performing Scenario: {best_scenario}\")\n",
    "print(f\"  - F2 Score: {best_f2_mean:.4f} ± {comparison_df[comparison_df['Scenario'].str.contains(best_scenario.split()[1])]['F2 Std'].values[0]:.4f}\")\n",
    "\n",
    "# Key findings\n",
    "print(\"\\nKey Findings:\")\n",
    "print(f\"  1. Scenario A (Baseline): F2 = {summary_a['F2 Mean']:.4f} ± {summary_a['F2 Std']:.4f}\")\n",
    "print(f\"  2. Scenario B (Clustering): F2 = {summary_b['F2 Mean']:.4f} ± {summary_b['F2 Std']:.4f}\")\n",
    "print(f\"  3. Scenario C (Model-based): F2 = {summary_c['F2 Mean']:.4f} ± {summary_c['F2 Std']:.4f}\")\n",
    "\n",
    "if p_value_ac < 0.05:\n",
    "    print(f\"\\n  ✓ Scenario C is STATISTICALLY SIGNIFICANTLY better than Scenario A (p = {p_value_ac:.4f})\")\n",
    "else:\n",
    "    print(f\"\\n  ✓ Scenario C is STATISTICALLY EQUIVALENT to Scenario A (p = {p_value_ac:.4f})\")\n",
    "\n",
    "if p_value_bc < 0.05:\n",
    "    print(f\"  ✓ Scenario C is STATISTICALLY SIGNIFICANTLY better than Scenario B (p = {p_value_bc:.4f})\")\n",
    "else:\n",
    "    print(f\"  ✓ Scenario C performance compared to Scenario B (p = {p_value_bc:.4f})\")\n",
    "\n",
    "print(\"\\n\" + \"=\"*80)\n"
]

# Replace cell 22 with enhanced version
nb['cells'][22] = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': enhanced_cell_22_source
}

print(f"\nEnhanced Cell 22 with comprehensive comparison")

# Write updated notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n{'='*80}")
print("✓ NOTEBOOK RESTRUCTURED SUCCESSFULLY")
print(f"{'='*80}")
print(f"Total cells: {len(nb['cells'])}")
print(f"\nStructure:")
print(f"  - Cells 1-13: Setup, data loading, model definitions")
print(f"  - Cells 14-18: Three scenario functions (A, B, C)")
print(f"  - Cell 19: Cross-validation section header")
print(f"  - Cell 20: 5-fold cross-validation code")
print(f"  - Cell 21: Budget analysis markdown")
print(f"  - Cell 22: ENHANCED comprehensive results aggregation & comparison")
print(f"\nRemoved:")
print(f"  - Old 'Approach 1 & 2' cells (29 cells removed)")
print(f"\nAdded to Cell 22:")
print(f"  ✓ Results aggregation across 5 folds")
print(f"  ✓ Mean ± Std calculations")
print(f"  ✓ Statistical significance tests (paired t-tests)")
print(f"  ✓ Box plot visualizations")
print(f"  ✓ Bar chart with error bars")
print(f"  ✓ Detailed conclusions")
print(f"  ✓ Results export to CSV and JSON")
print(f"\n{'='*80}")
