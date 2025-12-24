import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the import cell and fix it
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'from statsmodels.stats.contingency_tables import mcnemar' in source and 'torch.cuda.get_device_name' in source:
            # This is the problematic cell - the imports were appended without proper newlines
            if isinstance(cell['source'], list):
                # Find where the imports were incorrectly added
                for i, line in enumerate(cell['source']):
                    if 'torch.cuda.get_device_name(0)")from statsmodels' in line:
                        # Split this line properly
                        cell['source'][i] = '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
                        # Add the imports as separate lines after
                        cell['source'].insert(i+1, '\n')
                        cell['source'].insert(i+2, 'from statsmodels.stats.contingency_tables import mcnemar\n')
                        cell['source'].insert(i+3, 'from sklearn.calibration import calibration_curve\n')
                        print(f"Fixed cell {cell_idx}: Separated imports properly")
                        break
            break

# Save fixed notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Import syntax error fixed!")
