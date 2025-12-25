import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the confusion matrix cell
fixed = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'confusion_matrix' in ''.join(cell['source']):
        # Look for the problematic set_title line
        for j, line in enumerate(cell['source']):
            if 'axes[idx].set_title' in line and 'label' in line:
                # Check if this line or the next contains the error
                if not line.rstrip().endswith(')') and j + 1 < len(cell['source']):
                    # Multi-line f-string issue - combine lines
                    next_line = cell['source'][j + 1].strip()
                    # Replace with single-line version
                    cell['source'][j] = "    axes[idx].set_title(f'{label} - Fold 1', fontsize=12, fontweight='bold')\n"
                    # Remove the continuation line if it exists
                    if '(Fold 1)' in next_line or 'fontsize' in next_line:
                        cell['source'][j + 1] = ''
                    fixed = True
                    print(f"Fixed cell {i}, line {j}")
                    break

        # Clean up empty lines
        if fixed:
            cell['source'] = [line for line in cell['source'] if line.strip() or line == '\n']
            break

if not fixed:
    # Try alternative approach - search for the exact pattern
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell['source'])
            if "f'{label}\n(Fold 1)'" in source_text or "f'{label}\\n(Fold 1)'" in source_text:
                # Replace in the joined source
                new_source = source_text.replace("f'{label}\n(Fold 1)'", "f'{label} - Fold 1'")
                new_source = new_source.replace("f'{label}\\n(Fold 1)'", "f'{label} - Fold 1'")
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                                 for i, line in enumerate(cell['source'])]
                fixed = True
                print(f"Fixed cell {i} using alternative method")
                break

# Save
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

if fixed:
    print("Syntax error fixed successfully!")
else:
    print("Could not find the problematic line - it may already be fixed")
