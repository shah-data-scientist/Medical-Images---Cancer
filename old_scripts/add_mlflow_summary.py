"""
Add MLflow summary and viewing instructions
"""
import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

# Find conclusion section to insert before it
conclusion_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 10. Conclusion' in ''.join(cell['source']):
        conclusion_idx = i
        break

if conclusion_idx:
    mlflow_summary_md = """## 9.8 MLflow Experiment Tracking Summary

### Viewing Your Runs

**Expected MLflow Structure**:
- **5 Parent Runs**: One per fold (Fold_1, Fold_2, ..., Fold_5)
- **15 Child Runs**: 3 scenarios per fold (ScenarioA_Fold1, ScenarioB_Fold1, ScenarioC_Fold1, etc.)
- **Total**: 20 runs (5 parent + 15 nested children)

**To View in MLflow UI**:
1. Open terminal and run: `mlflow ui`
2. Navigate to: http://localhost:5000
3. Click on experiment: **BrainScanAI_SemiSupervised**
4. **Expand parent runs** (click arrow next to Fold_1, etc.) to see nested scenario runs

**Tip**: If you only see 5 runs, they are the parent fold runs. Click the arrow/expand icon to reveal the 3 scenario runs within each fold."""

    mlflow_summary_code = """# MLflow Runs Summary
import mlflow

# Get the experiment
experiment = mlflow.get_experiment_by_name("BrainScanAI_SemiSupervised")

if experiment:
    print("="*80)
    print("MLFLOW EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\\nExperiment Name: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")

    # Get all runs for this experiment
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    print(f"\\nTotal Runs: {len(runs)}")

    # Count parent and nested runs
    parent_runs = runs[runs['tags.mlflow.parentRunId'].isna()]
    nested_runs = runs[~runs['tags.mlflow.parentRunId'].isna()]

    print(f"  - Parent Runs (Folds): {len(parent_runs)}")
    print(f"  - Nested Runs (Scenarios): {len(nested_runs)}")

    print("\\n" + "="*80)
    print("RUN DETAILS")
    print("="*80)

    # Show parent runs with their children
    for idx, parent_run in parent_runs.iterrows():
        parent_run_id = parent_run['run_id']
        parent_name = parent_run['tags.mlflow.runName']

        print(f"\\n{parent_name} (Run ID: {parent_run_id[:8]}...)")

        # Find child runs
        children = runs[runs['tags.mlflow.parentRunId'] == parent_run_id]

        for child_idx, child_run in children.iterrows():
            child_name = child_run['tags.mlflow.runName']
            scenario = child_run['params.scenario'] if 'params.scenario' in child_run else 'Unknown'

            print(f"  └─ {child_name}")
            print(f"      Scenario: {scenario}")

            # Show key metrics if available
            if 'metrics.train_loss' in child_run:
                print(f"      Final Train Loss: {child_run.get('metrics.train_loss', 'N/A'):.4f}")
            if 'metrics.val_acc' in child_run:
                print(f"      Final Val Acc: {child_run.get('metrics.val_acc', 'N/A'):.4f}")

    print("\\n" + "="*80)
    print("HOW TO VIEW IN MLflow UI")
    print("="*80)
    print("\\n1. Run in terminal: mlflow ui")
    print("2. Open browser: http://localhost:5000")
    print("3. Click experiment: BrainScanAI_SemiSupervised")
    print("4. Expand parent runs (Fold_1, Fold_2, ...) to see nested scenarios")
    print("5. Compare metrics across runs using the 'Compare' button")
    print("6. Visualize training curves in the 'Charts' tab")
    print("\\n" + "="*80)

else:
    print("Experiment 'BrainScanAI_SemiSupervised' not found.")
    print("Make sure the notebook has been executed first.")"""

    # Insert before conclusion
    nb['cells'].insert(conclusion_idx, {
        'cell_type': 'markdown',
        'metadata': {},
        'source': mlflow_summary_md.split('\n')
    })

    nb['cells'].insert(conclusion_idx + 1, {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': mlflow_summary_code.split('\n')
    })

    print(f"Added MLflow summary at cells {conclusion_idx} and {conclusion_idx + 1}")
    print(f"Total cells now: {len(nb['cells'])}")

    # Save
    json.dump(nb, open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8'), indent=2)
    print("\nMLflow summary added!")
    print("\nThis cell will:")
    print("  - Show total runs (should be 20: 5 parent + 15 nested)")
    print("  - List all folds and their scenario runs")
    print("  - Provide instructions for MLflow UI")
else:
    print("Could not find conclusion section")
