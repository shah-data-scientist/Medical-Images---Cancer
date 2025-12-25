import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix the f-string syntax error
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            # Fix the problematic f-string with newline
            if "f'{label}\\n(Fold 1)'" in line or "f'{label}\n(Fold 1)'" in line:
                cell['source'][i] = line.replace("f'{label}\\n(Fold 1)'", "f'{label} - Fold 1'")
                cell['source'][i] = cell['source'][i].replace("f'{label}\n(Fold 1)'", "f'{label} - Fold 1'")
                print(f"Fixed line: {line.strip()}")

# Save
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\nSyntax error fixed!")
