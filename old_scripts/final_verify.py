import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

print("="*60)
print("NOTEBOOK 3 FINAL VERIFICATION")
print("="*60)
print(f"\nTotal cells: {len(nb['cells'])}")

code_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Code cells: {len(code_cells)}")
print(f"Markdown cells: {len(nb['cells']) - len(code_cells)}")

# Check for any remaining issues
issues = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'prob_supervised' in source:
            issues.append(f"Cell {i}: OLD variable 'prob_supervised'")
        if 'prob_semisup' in source:
            issues.append(f"Cell {i}: OLD variable 'prob_semisup'")
        if 'y_prob_sup[:, 1]' in source or 'y_prob_semi[:, 1]' in source:
            issues.append(f"Cell {i}: Incorrect array indexing")

if issues:
    print(f"\nISSUES FOUND ({len(issues)}):")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\nNO ISSUES FOUND - All cells verified!")

print("\n" + "="*60)
print("READY FOR EXECUTION")
print("="*60)
