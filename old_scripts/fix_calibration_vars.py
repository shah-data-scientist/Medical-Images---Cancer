import json

# Load Notebook 3
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the calibration cell and fix variable names
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'CALIBRATION ANALYSIS' in source_text and 'prob_supervised' in source_text:
            # Replace incorrect variable names with correct ones
            for i, line in enumerate(cell['source']):
                line = line.replace('prob_supervised', 'y_prob_sup')
                line = line.replace('prob_semisup', 'y_prob_semi')
                line = line.replace('y_test,', 'y_true_sup,')
                cell['source'][i] = line

            print("Fixed calibration cell variable names:")
            print("  - prob_supervised -> y_prob_sup")
            print("  - prob_semisup -> y_prob_semi")
            print("  - y_test -> y_true_sup")
            break

# Save the updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("\nCalibration cell fixed successfully!")
