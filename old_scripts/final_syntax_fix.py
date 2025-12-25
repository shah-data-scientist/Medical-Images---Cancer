import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

# Find and fix cell 26 (confusion matrix cell)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'confusion_matrix' in ''.join(cell['source']):
        # Find the problematic line
        for j, line in enumerate(cell['source']):
            if 'axes[idx].set_title' in line and '\n(Fold 1)' in line:
                # Replace with fixed version
                cell['source'][j] = "    axes[idx].set_title(f'{label} - Fold 1', fontsize=12, fontweight='bold')\n"
                print(f"Fixed cell {i}, line {j}")
                break

json.dump(nb, open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8'), indent=2)
print("Syntax error fixed!")
