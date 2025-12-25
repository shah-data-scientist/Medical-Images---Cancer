#!/usr/bin/env python3
"""
Script to recreate missing cells in 3_semi_supervised_learning.ipynb
Adds Scenario A, B, C functions and cross-validation code.
"""

import json
import sys
from pathlib import Path


def create_code_cell(source_lines):
    """Create a code cell with given source lines."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }


def create_markdown_cell(source_lines):
    """Create a markdown cell with given source lines."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }


def get_scenario_a_cell():
    """Return Scenario A function cell."""
    source = [
        "def scenario_a_fully_supervised(train_idx, val_idx, fold):\n",
        "    \"\"\"Scenario A: Fully Supervised (Baseline)\"\"\"\n",
        "    with mlflow.start_run(run_name=f\"ScenarioA_Fold{fold}\", nested=True):\n",
        "        mlflow.log_params({\n",
        "            \"scenario\": \"Fully_Supervised\",\n",
        "            \"fold\": fold,\n",
        "            \"train_size\": len(train_idx),\n",
        "            \"val_size\": len(val_idx),\n",
        "            \"input_dim\": 50,\n",
        "            \"hidden_dim\": 128,\n",
        "            \"dropout\": 0.5,\n",
        "            \"learning_rate\": 0.001,\n",
        "            \"weight_decay\": 0.01\n",
        "        })\n",
        "        \n",
        "        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])\n",
        "        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])\n",
        "        \n",
        "        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)\n",
        "        val_loader = DataLoader(val_dataset, batch_size=16)\n",
        "        \n",
        "        model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)\n",
        "        criterion = nn.CrossEntropyLoss()\n",
        "        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)\n",
        "        \n",
        "        model, history = train_model(model, train_loader, val_loader, criterion, optimizer)\n",
        "        \n",
        "        for epoch, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):\n",
        "            mlflow.log_metrics({\n",
        "                'train_loss': tl,\n",
        "                'val_loss': vl,\n",
        "                'val_acc': va\n",
        "            }, step=epoch)\n",
        "        \n",
        "        return model\n"
    ]
    return create_code_cell(source)


def get_scenario_b_markdown_cell():
    """Return Scenario B markdown cell."""
    source = [
        "### Scenario B: Semi-Supervised (Clustering-based)\n",
        "\n",
        "**Two-phase training with ALL weak labels (no filtering):**\n",
        "\n",
        "**Phase 1: Pre-train on ALL Weak Labels**\n",
        "- Uses **ALL 1,406 weak labels** from K-means clustering\n",
        "- Quality: ~82% agreement with ground truth\n",
        "- **Fixed 20 epochs** (no early stopping)\n",
        "\n",
        "**Phase 2: Fine-tune on Strong Labels**\n",
        "- Uses clean expert-labeled data (56 samples per fold)\n",
        "- With validation monitoring for early stopping\n"
    ]
    return create_markdown_cell(source)


def get_scenario_b_code_cell():
    """Return Scenario B function cell with CRITICAL FIX."""
    source = [
        "def scenario_b_clustering_semisup(train_idx, val_idx, fold):\n",
        "    \"\"\"Scenario B: Semi-Supervised with ALL Clustering Weak Labels\"\"\"\n",
        "    with mlflow.start_run(run_name=f\"ScenarioB_Fold{fold}\", nested=True):\n",
        "        mlflow.log_params({\n",
        "            \"scenario\": \"SemiSup_Clustering_AllLabels\",\n",
        "            \"fold\": fold,\n",
        "            \"filtering\": \"None_All_Labels\"\n",
        "        })\n",
        "        \n",
        "        # CRITICAL FIX: Filter to unlabeled data only\n",
        "        unlabeled_weak_labels = weak_labels_df[weak_labels_df['split'] == 'unlabeled'].copy()\n",
        "        weak_features = unlabeled_pca[:len(unlabeled_weak_labels)]\n",
        "        weak_labels = unlabeled_weak_labels['weak_label_kmeans'].values\n",
        "        \n",
        "        cluster_0_count = (weak_labels == 0).sum()\n",
        "        cluster_1_count = (weak_labels == 1).sum()\n",
        "        \n",
        "        print(f\"  Using {len(weak_labels)} weak labels\")\n",
        "        print(f\"    - Cluster 0: {cluster_0_count}, Cluster 1: {cluster_1_count}\")\n",
        "        \n",
        "        weak_dataset = FeatureDataset(weak_features, weak_labels)\n",
        "        weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True)\n",
        "        \n",
        "        model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)\n",
        "        criterion = nn.CrossEntropyLoss()\n",
        "        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)\n",
        "        \n",
        "        # Pre-training (20 epochs)\n",
        "        print(f\"  Phase 1: Pre-training...\")\n",
        "        for epoch in range(20):\n",
        "            model.train()\n",
        "            for features, labels in weak_loader:\n",
        "                features, labels = features.to(device), labels.to(device)\n",
        "                optimizer.zero_grad()\n",
        "                outputs = model(features)\n",
        "                loss = criterion(outputs, labels)\n",
        "                loss.backward()\n",
        "                optimizer.step()\n",
        "        \n",
        "        print(f\"  Phase 2: Fine-tuning...\")\n",
        "        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])\n",
        "        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])\n",
        "        \n",
        "        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)\n",
        "        val_loader = DataLoader(val_dataset, batch_size=16)\n",
        "        \n",
        "        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)\n",
        "        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)\n",
        "        \n",
        "        return model\n"
    ]
    return create_code_cell(source)


def get_scenario_c_markdown_cell():
    """Return Scenario C markdown cell."""
    source = [
        "### Scenario C: Semi-Supervised (Model-based)\n",
        "\n",
        "**Three-phase training:**\n",
        "- Phase 1: Train initial model on labeled data\n",
        "- Phase 2: Generate high-confidence pseudo-labels (≥90%)\n",
        "- Phase 3: Retrain on combined dataset\n"
    ]
    return create_markdown_cell(source)


def get_scenario_c_code_cell():
    """Return Scenario C function cell."""
    source = [
        "def scenario_c_model_semisup(train_idx, val_idx, fold):\n",
        "    \"\"\"Scenario C: Semi-Supervised with Model-based Pseudo-labels\"\"\"\n",
        "    with mlflow.start_run(run_name=f\"ScenarioC_Fold{fold}\", nested=True):\n",
        "        mlflow.log_param(\"scenario\", \"SemiSup_ModelBased\")\n",
        "        \n",
        "        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])\n",
        "        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])\n",
        "        \n",
        "        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)\n",
        "        val_loader = DataLoader(val_dataset, batch_size=16)\n",
        "        \n",
        "        initial_model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)\n",
        "        criterion = nn.CrossEntropyLoss()\n",
        "        optimizer = optim.Adam(initial_model.parameters(), lr=0.001, weight_decay=0.01)\n",
        "        \n",
        "        initial_model, _ = train_model(initial_model, train_loader, val_loader, criterion, optimizer, epochs=30)\n",
        "        print(f\"  Phase 1: Initial model trained\")\n",
        "        \n",
        "        # Generate pseudo-labels\n",
        "        unlabeled_dataset = FeatureDataset(unlabeled_pca, np.zeros(len(unlabeled_pca)))\n",
        "        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32)\n",
        "        \n",
        "        initial_model.eval()\n",
        "        pseudo_labels = []\n",
        "        pseudo_confidences = []\n",
        "        \n",
        "        with torch.no_grad():\n",
        "            for features, _ in unlabeled_loader:\n",
        "                features = features.to(device)\n",
        "                outputs = initial_model(features)\n",
        "                probs = torch.softmax(outputs, dim=1)\n",
        "                confidence, predicted = probs.max(1)\n",
        "                pseudo_labels.extend(predicted.cpu().numpy())\n",
        "                pseudo_confidences.extend(confidence.cpu().numpy())\n",
        "        \n",
        "        pseudo_labels = np.array(pseudo_labels)\n",
        "        pseudo_confidences = np.array(pseudo_confidences)\n",
        "        \n",
        "        high_conf_mask = pseudo_confidences >= 0.9\n",
        "        high_conf_features = unlabeled_pca[high_conf_mask]\n",
        "        high_conf_labels = pseudo_labels[high_conf_mask]\n",
        "        \n",
        "        print(f\"  Phase 2: Generated {high_conf_mask.sum()} high-conf pseudo-labels\")\n",
        "        \n",
        "        if high_conf_mask.sum() > 0:\n",
        "            combined_features = np.vstack([all_labeled_pca[train_idx], high_conf_features])\n",
        "            combined_labels = np.hstack([all_labeled_labels[train_idx], high_conf_labels])\n",
        "            \n",
        "            combined_dataset = FeatureDataset(combined_features, combined_labels)\n",
        "            combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=True)\n",
        "            \n",
        "            final_model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)\n",
        "            optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.01)\n",
        "            \n",
        "            final_model, history = train_model(final_model, combined_loader, val_loader, criterion, optimizer)\n",
        "            print(f\"  Phase 3: Retrained on {len(combined_labels)} samples\")\n",
        "        else:\n",
        "            final_model = initial_model\n",
        "        \n",
        "        return final_model\n"
    ]
    return create_code_cell(source)


def get_cross_validation_cell():
    """Return 5-fold cross-validation cell."""
    source = [
        "print(\"=\"*80)\n",
        "print(\"STARTING 5-FOLD CROSS-VALIDATION\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)\n",
        "results = {'scenario_a': [], 'scenario_b': [], 'scenario_c': []}\n",
        "\n",
        "test_loader = DataLoader(FeatureDataset(test_pca, test_labels), batch_size=16)\n",
        "\n",
        "for fold, (train_idx, val_idx) in enumerate(skf.split(all_labeled_pca, all_labeled_labels), 1):\n",
        "    print(f\"\\nFOLD {fold}/5\")\n",
        "    \n",
        "    with mlflow.start_run(run_name=f\"Fold_{fold}\"):\n",
        "        # Scenario A\n",
        "        print(\"[1/3] Scenario A...\")\n",
        "        model_a = scenario_a_fully_supervised(train_idx, val_idx, fold)\n",
        "        metrics_a, preds_a, labels_a, probs_a = evaluate_model(model_a, test_loader)\n",
        "        results['scenario_a'].append((metrics_a, preds_a, probs_a))\n",
        "        print(f\"  F2: {metrics_a['f2']:.4f}\")\n",
        "        \n",
        "        # Scenario B\n",
        "        print(\"[2/3] Scenario B...\")\n",
        "        model_b = scenario_b_clustering_semisup(train_idx, val_idx, fold)\n",
        "        metrics_b, preds_b, labels_b, probs_b = evaluate_model(model_b, test_loader)\n",
        "        results['scenario_b'].append((metrics_b, preds_b, probs_b))\n",
        "        print(f\"  F2: {metrics_b['f2']:.4f}\")\n",
        "        \n",
        "        # Scenario C\n",
        "        print(\"[3/3] Scenario C...\")\n",
        "        model_c = scenario_c_model_semisup(train_idx, val_idx, fold)\n",
        "        metrics_c, preds_c, labels_c, probs_c = evaluate_model(model_c, test_loader)\n",
        "        results['scenario_c'].append((metrics_c, preds_c, probs_c))\n",
        "        print(f\"  F2: {metrics_c['f2']:.4f}\")\n",
        "\n",
        "print(\"\\nCROSS-VALIDATION COMPLETE\")\n"
    ]
    return create_code_cell(source)


def get_budget_analysis_markdown_cell():
    """Return budget analysis markdown cell."""
    source = [
        "## Budget Analysis\n",
        "\n",
        "Comparing scenarios across different budget levels:\n",
        "- **Scenario A**: Fully supervised baseline\n",
        "- **Scenario B**: Semi-supervised with clustering weak labels\n",
        "- **Scenario C**: Semi-supervised with model-based pseudo-labels\n"
    ]
    return create_markdown_cell(source)


def get_budget_analysis_code_cell():
    """Return budget analysis code cell."""
    source = [
        "# Calculate average metrics across folds\n",
        "import pandas as pd\n",
        "\n",
        "def average_metrics(results_list):\n",
        "    \"\"\"Average metrics across folds.\"\"\"\n",
        "    avg_metrics = {}\n",
        "    for key in results_list[0][0].keys():\n",
        "        avg_metrics[key] = np.mean([r[0][key] for r in results_list])\n",
        "    return avg_metrics\n",
        "\n",
        "avg_a = average_metrics(results['scenario_a'])\n",
        "avg_b = average_metrics(results['scenario_b'])\n",
        "avg_c = average_metrics(results['scenario_c'])\n",
        "\n",
        "comparison_df = pd.DataFrame({\n",
        "    'Scenario': ['A: Fully Supervised', 'B: Clustering Semi-Sup', 'C: Model Semi-Sup'],\n",
        "    'F2 Score': [avg_a['f2'], avg_b['f2'], avg_c['f2']],\n",
        "    'Recall': [avg_a['recall'], avg_b['recall'], avg_c['recall']],\n",
        "    'Precision': [avg_a['precision'], avg_b['precision'], avg_c['precision']],\n",
        "    'Accuracy': [avg_a['accuracy'], avg_b['accuracy'], avg_c['accuracy']]\n",
        "})\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"FINAL COMPARISON (Averaged across 5 folds)\")\n",
        "print(\"=\"*80)\n",
        "print(comparison_df.to_string(index=False))\n"
    ]
    return create_code_cell(source)


def find_insertion_point(cells, search_terms, start_from=0):
    """
    Find the index where to insert new cells.
    Searches for cells containing any of the search terms.
    Returns the index after the found cell.
    """
    for i in range(start_from, len(cells)):
        cell = cells[i]
        if cell['cell_type'] == 'code' and cell['source']:
            source_text = ''.join(cell['source'])
            for term in search_terms:
                if term in source_text:
                    return i + 1
    return -1


def main():
    notebook_path = Path(r"c:\Users\shahu\Documents\OneDrive\OPEN CLASSROOMS\PROJET 7\Medical Images - Cancer\3_semi_supervised_learning.ipynb")

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)

    print(f"Reading notebook: {notebook_path}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    original_count = len(cells)
    print(f"Original cell count: {original_count}")

    # Strategy: Find specific insertion points based on content analysis
    # We need to insert after the train_model and evaluate_model definitions

    # Look for a cell that might contain training functions or evaluation
    # Based on the cell listing, we should insert after cell 13 (training functions)

    # Insert Scenario A after training/evaluation functions (around cell 14)
    insertion_idx = 14

    print(f"\nInserting Scenario A at cell {insertion_idx}")
    cells.insert(insertion_idx, get_scenario_a_cell())
    insertion_idx += 1

    print(f"Inserting Scenario B markdown at cell {insertion_idx}")
    cells.insert(insertion_idx, get_scenario_b_markdown_cell())
    insertion_idx += 1

    print(f"Inserting Scenario B code at cell {insertion_idx}")
    cells.insert(insertion_idx, get_scenario_b_code_cell())
    insertion_idx += 1

    print(f"Inserting Scenario C markdown at cell {insertion_idx}")
    cells.insert(insertion_idx, get_scenario_c_markdown_cell())
    insertion_idx += 1

    print(f"Inserting Scenario C code at cell {insertion_idx}")
    cells.insert(insertion_idx, get_scenario_c_code_cell())
    insertion_idx += 1

    # Insert a markdown header for the cross-validation section
    cv_markdown = create_markdown_cell([
        "## Cross-Validation Experiments\n",
        "\n",
        "Running 5-fold stratified cross-validation for all three scenarios.\n"
    ])
    print(f"Inserting CV header at cell {insertion_idx}")
    cells.insert(insertion_idx, cv_markdown)
    insertion_idx += 1

    print(f"Inserting 5-fold cross-validation at cell {insertion_idx}")
    cells.insert(insertion_idx, get_cross_validation_cell())
    insertion_idx += 1

    print(f"Inserting budget analysis markdown at cell {insertion_idx}")
    cells.insert(insertion_idx, get_budget_analysis_markdown_cell())
    insertion_idx += 1

    print(f"Inserting budget analysis code at cell {insertion_idx}")
    cells.insert(insertion_idx, get_budget_analysis_code_cell())

    # Update notebook
    notebook['cells'] = cells
    new_count = len(cells)
    print(f"\nNew cell count: {new_count} (added {new_count - original_count} cells)")

    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.backup')
    print(f"\nCreating backup at: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    # Save updated notebook
    print(f"Saving updated notebook: {notebook_path}")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print("\n" + "="*80)
    print("SUCCESS! Notebook updated with missing cells.")
    print("="*80)
    print(f"\nCells added:")
    print("  1. Scenario A function (fully supervised)")
    print("  2. Scenario B markdown + function (clustering semi-supervised)")
    print("  3. Scenario C markdown + function (model-based semi-supervised)")
    print("  4. Cross-validation header")
    print("  5. 5-fold cross-validation code")
    print("  6. Budget analysis markdown")
    print("  7. Budget analysis code")
    print(f"\nBackup saved to: {backup_path}")
    print("\nCRITICAL FIX APPLIED:")
    print("  - Scenario B now filters to unlabeled data only")
    print("  - This prevents index errors when accessing unlabeled_pca")


if __name__ == "__main__":
    main()
