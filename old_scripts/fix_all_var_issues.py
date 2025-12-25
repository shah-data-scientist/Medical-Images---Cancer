import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed_count = 0

# Fix all cells with incorrect variable names
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])

        # Check if this cell has the problematic variables
        if 'prob_supervised' in source_text or 'prob_semisup' in source_text or ('y_test' in source_text and 'calibration' in source_text.lower()):
            for j, line in enumerate(cell['source']):
                original = line
                line = line.replace('prob_supervised', 'y_prob_sup')
                line = line.replace('prob_semisup', 'y_prob_semi')

                # Only replace y_test in calibration context (not in other contexts)
                if 'calibration_curve' in source_text or 'calculate_ece' in source_text:
                    line = line.replace('y_test,', 'y_true_sup,')
                    line = line.replace('y_test)', 'y_true_sup)')

                if line != original:
                    cell['source'][j] = line
                    fixed_count += 1

print(f"Fixed {fixed_count} lines with incorrect variable names")

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("All variable issues fixed in Notebook 3")
