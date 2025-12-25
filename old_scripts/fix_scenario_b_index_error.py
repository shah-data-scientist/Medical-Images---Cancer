"""
Fix index error in Scenario B:
The weak_labels.csv contains ALL data (1,506 rows including labeled),
but we need only unlabeled data (1,406 rows)
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix Scenario B function (Cell 17)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'def scenario_b_clustering_semisup' in ''.join(cell.get('source', [])):

        # Find the problematic lines and fix them
        source_text = ''.join(cell['source'])

        # Replace the weak label loading to filter only unlabeled data
        old_code = '''        # Phase 1: Pre-train on ALL weak labels (no filtering)
        # Use weak_label_kmeans column which contains all cluster assignments
        weak_features = unlabeled_pca[:len(weak_labels_df)]
        weak_labels = weak_labels_df['weak_label_kmeans'].values'''

        new_code = '''        # Phase 1: Pre-train on ALL weak labels (no filtering)
        # Use weak_label_kmeans column which contains all cluster assignments
        # Filter to get only unlabeled data (exclude the 100 labeled samples)
        unlabeled_weak_labels = weak_labels_df[weak_labels_df['split'] == 'unlabeled'].copy()
        weak_features = unlabeled_pca[:len(unlabeled_weak_labels)]
        weak_labels = unlabeled_weak_labels['weak_label_kmeans'].values'''

        updated_source = source_text.replace(old_code, new_code)

        cell['source'] = updated_source.split('\n')
        print(f"[OK] Cell {i}: Fixed Scenario B to filter only unlabeled weak labels")
        break

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\n[OK] Index error fixed!")
print("Scenario B will now correctly use only the 1,406 unlabeled samples.")
