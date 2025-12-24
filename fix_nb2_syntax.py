import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix cell 32 (index 31)
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "'n_pca_components': int(features_pca_50.shape[1])" in source and "'confidence_threshold'" in source:
            # Fix the missing comma
            if isinstance(cell['source'], list):
                for i, line in enumerate(cell['source']):
                    if "'n_pca_components': int(features_pca_50.shape[1])" in line and not line.rstrip().endswith(','):
                        cell['source'][i] = line.rstrip() + ',\n'
                        print(f"Fixed cell {cell_idx}: Added comma after n_pca_components")
                        break
            break

# Save fixed notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Syntax error fixed!")
