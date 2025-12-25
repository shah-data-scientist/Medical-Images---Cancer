"""
Add index/table of contents at the beginning of Notebook 3
"""
import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

# Create table of contents
toc_markdown = """## Table of Contents

### Part I: Introduction & Setup
- [Overview](#Overview)
- [Objectives](#Objectives)
  - Training & Evaluation
  - Business Analysis
- [Evaluation Metrics](#Evaluation-Metrics) - Why F-beta Score (β=2)?
- [Methodology](#Methodology)
- [Setup & Imports](#1.-Load-Data)

### Part II: Data Preparation
- [1. Load Data](#1.-Load-Data) - Split-specific features, PCA, weak labels
- [1.5 Weak Labeling Strategy](#1.5-Weak-Labeling-Strategy)
  - How were weak labels generated?
  - Weak label quality metrics
  - Usage strategy by scenario

### Part III: Model Architecture & Training
- [2. Model Architecture](#2.-Model-Architecture) - 2-layer neural network
- [3. Dataset and DataLoader](#3.-Dataset-and-DataLoader)
- [4. Training and Evaluation Functions](#4.-Training-and-Evaluation-Functions)

### Part IV: Three Training Scenarios
- [5. Scenario Implementations](#5.-Scenario-Implementations)
  - **Scenario A**: Fully Supervised (baseline)
  - **Scenario B**: Semi-Supervised (clustering-based weak labels)
  - **Scenario C**: Semi-Supervised (model-based pseudo-labels) ⭐ NEW
- [6. 5-Fold Cross-Validation](#6.-5-Fold-Cross-Validation) - Run all scenarios

### Part V: Results & Analysis
- [7. Results Aggregation](#7.-Results-Aggregation-and-Visualization) - Mean ± Std across folds
- [8. Statistical Comparison](#8.-Statistical-Comparison) - McNemar's Test
- [9. Visualizations](#9.-Visualization)
  - Metrics comparison (Recall, F2, Precision, Accuracy)
  - ROC Curves
  - Confusion Matrices
  - Training History

### Part VI: Business & Deployment
- [9.7 Budget Analysis](#9.7-Budget-Analysis-&-Scaling-Feasibility) - €5,000 labeling strategy
  - 4 budget scenarios comparison
  - Cost-effectiveness analysis
  - Data-driven recommendations
- [9.8 MLflow Tracking](#9.8-MLflow-Experiment-Tracking-Summary) - View experiment runs
- [9.5 Model Persistence](#9.5-Model-Persistence) - Save trained models

### Part VII: Conclusions
- [10. Conclusion](#10.-Conclusion)
  - Key findings
  - Performance summary
  - Next steps for production

---

**Quick Navigation Tips**:
- Click any section title to jump directly to it
- Total sections: 39 cells
- Estimated execution time: 30-50 minutes (5 folds × 3 scenarios)

---"""

# Insert TOC after the title cell (cell 0)
nb['cells'].insert(1, {
    'cell_type': 'markdown',
    'metadata': {},
    'source': toc_markdown.split('\n')
})

print(f"Added Table of Contents at cell 1")
print(f"Total cells now: {len(nb['cells'])}")

# Save
json.dump(nb, open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8'), indent=2)
print("\nTable of Contents added successfully!")
print("\nThe index includes:")
print("  - 7 major parts (Introduction → Conclusion)")
print("  - All key sections with navigation links")
print("  - Quick execution time estimate")
