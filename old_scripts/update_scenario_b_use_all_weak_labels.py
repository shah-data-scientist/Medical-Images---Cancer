"""
Update Scenario B to use ALL weak labels (no filtering)

Changes:
- Cell 6: Load weak_labels.csv (all 1,406 labels)
- Cell 16: Update markdown to reflect using all weak labels
- Cell 17: Update code to use all weak labels without filtering
- Cell 7: Update weak labeling strategy section
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ===================================================================
# CELL 6: Update data loading to use ALL weak labels
# ===================================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'FEATURES_DIR = Path' in ''.join(cell.get('source', [])):
        source_text = ''.join(cell['source'])

        # Replace the weak labels loading line
        updated_source = source_text.replace(
            "# Load weak labels\n# TOP 20% high-confidence weak labels (filtered in Notebook 2)\nweak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels_high_confidence.csv')",
            "# Load weak labels\n# ALL weak labels from K-means clustering (no filtering)\nweak_labels_df = pd.read_csv(FEATURES_DIR / 'weak_labels.csv')"
        )

        # Update the print statement
        updated_source = updated_source.replace(
            'print(f"Weak labels available: {len(weak_labels_df)}")',
            'print(f"Weak labels available (K-means, unfiltered): {len(weak_labels_df)}")'
        )

        cell['source'] = updated_source.split('\n')
        print(f"[OK] Cell {i} (Data Loading): Updated to load ALL weak labels from weak_labels.csv")
        break

# ===================================================================
# CELL 7: Update weak labeling strategy markdown
# ===================================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '## 1.5 Weak Labeling Strategy' in ''.join(cell.get('source', [])):
        # Update the Scenario B section
        source_text = ''.join(cell['source'])

        # Find and replace the Scenario B section
        old_scenario_b = '''**Scenario B (Semi-Supervised - Clustering)**: ⚠️ REQUIRES FILTERED DATA
- Uses: ~282 balanced weak labels from stratified K-means filtering
- **CRITICAL**: Must execute Notebook 2 with stratified filtering first!
- Without filtering: Uses all 1,406 noisy labels → Poor performance (F2 ~0.76)
- With filtering: Uses 282 clean labels → Better performance (F2 ~0.82-0.87)
- Phase 1: Pre-train on 282 high-confidence **balanced** weak labels
- Phase 2: Fine-tune on 60-70 strong labels
- **Key advantage**: Balanced pre-training prevents class bias'''

        new_scenario_b = '''**Scenario B (Semi-Supervised - Clustering)**: Uses ALL Weak Labels
- Uses: ALL 1,406 weak labels from K-means clustering (no filtering)
- Quality: ~82% agreement with ground truth (estimated 250+ noisy labels)
- Class distribution: Cluster 0: ~603, Cluster 1: ~803 (ratio 1:1.33)
- Phase 1: Pre-train on all 1,406 weak labels (noisy but comprehensive)
- Phase 2: Fine-tune on 60-70 strong labels
- **Experiment**: Compare quantity (all data) vs. quality (filtered data)'''

        updated_source = source_text.replace(old_scenario_b, new_scenario_b)

        cell['source'] = updated_source.split('\n')
        print(f"[OK] Cell {i} (Weak Labeling Strategy): Updated Scenario B description")
        break

# ===================================================================
# CELL 16: Update Scenario B markdown explanation
# ===================================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Scenario B: Semi-Supervised (Clustering-based)' in ''.join(cell.get('source', [])) and 'Two-phase training' in ''.join(cell.get('source', [])):

        new_markdown = '''### Scenario B: Semi-Supervised (Clustering-based)

**Two-phase training with ALL weak labels (no filtering):**

**Phase 1: Pre-train on ALL Weak Labels**
- Uses **ALL 1,406 weak labels** from K-means clustering (Notebook 2)
- Quality: ~82% agreement with ground truth (estimated ~250 noisy labels)
- Class distribution: Cluster 0: ~603 (42.9%), Cluster 1: ~803 (57.1%)
- **Fixed 20 epochs** (no early stopping to avoid overfitting to noisy labels)

**Phase 2: Fine-tune on Strong Labels**
- Uses clean expert-labeled data from train split (56 samples per fold)
- **With validation monitoring** for early stopping
- Refines the features learned in Phase 1
- Corrects errors from noisy weak labels

**Experiment Rationale:**
- **Quantity vs. Quality**: Does more data (1,406) outweigh label noise?
- Previous filtered approach (282 labels): F2 ~0.60 (underperformed)
- This unfiltered approach tests if comprehensive coverage helps
- Compare with model-based pseudo-labeling (Scenario C)

**Expected Trade-offs:**
- More pre-training data (1,406 vs 282)
- Higher label noise (18% vs 10-15%)
- Better coverage of data distribution
- Risk: Noise may hurt more than quantity helps'''

        cell['source'] = new_markdown.split('\n')
        print(f"[OK] Cell {i} (Scenario B Markdown): Updated to describe unfiltered approach")
        break

# ===================================================================
# CELL 17: Update Scenario B code
# ===================================================================
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'def scenario_b_clustering_semisup' in ''.join(cell.get('source', [])):

        new_code = '''def scenario_b_clustering_semisup(train_idx, val_idx, fold):
    """Scenario B: Semi-Supervised with ALL Clustering Weak Labels (No Filtering)"""
    with mlflow.start_run(run_name=f"ScenarioB_Fold{fold}", nested=True):
        mlflow.log_params({
            "scenario": "SemiSup_Clustering_AllLabels",
            "fold": fold,
            "train_size": len(train_idx),
            "filtering": "None_All_Labels"
        })

        # Phase 1: Pre-train on ALL weak labels (no filtering)
        # Use weak_label_kmeans column which contains all cluster assignments
        weak_features = unlabeled_pca[:len(weak_labels_df)]
        weak_labels = weak_labels_df['weak_label_kmeans'].values

        # Log weak label information
        cluster_0_count = (weak_labels == 0).sum()
        cluster_1_count = (weak_labels == 1).sum()
        balance_ratio = cluster_1_count / max(cluster_0_count, 1)

        mlflow.log_param("weak_labels_total", len(weak_labels))
        mlflow.log_param("weak_labels_cluster_0", int(cluster_0_count))
        mlflow.log_param("weak_labels_cluster_1", int(cluster_1_count))
        mlflow.log_param("balance_ratio", float(balance_ratio))

        print(f"  Using ALL {len(weak_labels)} weak labels (unfiltered K-means)")
        print(f"    - Cluster 0: {cluster_0_count} samples ({cluster_0_count/len(weak_labels)*100:.1f}%)")
        print(f"    - Cluster 1: {cluster_1_count} samples ({cluster_1_count/len(weak_labels)*100:.1f}%)")
        print(f"    - Balance ratio: 1:{balance_ratio:.2f}")

        # Create dataset with ALL weak labels
        weak_dataset = FeatureDataset(weak_features, weak_labels)
        weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True)

        # Pre-train model
        model = BrainTumorClassifier(input_dim=50, dropout=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        # Pre-training (fixed 20 epochs - no early stopping to avoid overfitting to noisy labels)
        print(f"  Phase 1: Pre-training on {len(weak_labels)} weak labels (including noise)...")
        for epoch in range(20):
            model.train()
            epoch_loss = 0
            for features, labels in weak_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Log pre-training loss
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(weak_loader)
                mlflow.log_metric("pretrain_loss", avg_loss, step=epoch)

        print(f"  Phase 1 complete: Model pre-trained on all weak labels")

        # Phase 2: Fine-tune on strong labels WITH VALIDATION
        print(f"  Phase 2: Fine-tuning on {len(train_idx)} strong labels...")
        train_dataset = FeatureDataset(all_labeled_pca[train_idx], all_labeled_labels[train_idx])
        val_dataset = FeatureDataset(all_labeled_pca[val_idx], all_labeled_labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Fine-tune with lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

        print(f"  Phase 2 complete: Model fine-tuned on strong labels")

        # Log fine-tuning metrics
        for epoch, (tl, vl, va) in enumerate(zip(history['train_loss'], history['val_loss'], history['val_acc'])):
            mlflow.log_metrics({
                'finetune_train_loss': tl,
                'finetune_val_loss': vl,
                'finetune_val_acc': va
            }, step=epoch)

        return model

print("Scenario B function defined")'''

        cell['source'] = new_code.split('\n')
        print(f"[OK] Cell {i} (Scenario B Code): Updated to use ALL weak labels")
        break

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("SCENARIO B UPDATE COMPLETE")
print("="*80)
print("\nChanges applied:")
print("  1. Cell 6: Load weak_labels.csv (ALL 1,406 labels)")
print("  2. Cell 7: Updated Scenario B strategy description")
print("  3. Cell 16: Updated markdown to explain unfiltered approach")
print("  4. Cell 17: Updated code to use all weak labels without filtering")
print("\nScenario B now uses:")
print("  - ALL 1,406 weak labels from K-means clustering")
print("  - No confidence filtering applied")
print("  - Class distribution: ~603 vs ~803 (1:1.33 ratio)")
print("\nExpected impact:")
print("  - More pre-training data (5x more than filtered)")
print("  - Higher label noise (~250 wrong labels vs ~28-42)")
print("  - Test whether quantity compensates for noise")
print("\n" + "="*80)
