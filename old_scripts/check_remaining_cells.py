import json

nb = json.load(open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8'))

print("Remaining cells after calibration (cell 38):\n")
for i, cell in enumerate(nb['cells'][38:], 38):
    cell_type = cell['cell_type']
    content = ''.join(cell['source'])[:150]

    print(f"Cell {i} ({cell_type}):")
    if cell_type == 'markdown':
        first_line = content.split('\n')[0]
        print(f"  {first_line}")
    else:
        # Check for potential issues
        issues = []
        if 'prob_supervised' in content or 'prob_semisup' in content:
            issues.append("OLD variable names still present")
        if '[:, 1]' in content and ('y_prob' in content):
            issues.append("Incorrect array indexing")

        if issues:
            print(f"  ⚠️ Issues: {', '.join(issues)}")
        else:
            print(f"  ✓ Looks good")

        print(f"  First line: {content.split(chr(10))[0][:80]}")
    print()
