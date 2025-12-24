import json

# Update Notebook 2 - summary_stats
print("Updating Notebook 2...")
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "'confidence_threshold': 0.9" in line:
                cell['source'][i] = line.replace("'confidence_threshold': 0.9", "'confidence_threshold': 0.266")
                print(f"  ✓ Updated summary_stats in code cell")

with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1)

# Update Notebook 3 - markdown reference
print("\nUpdating Notebook 3...")
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb3 = json.load(f)

for cell in nb3['cells']:
    if cell['cell_type'] == 'markdown':
        for i, line in enumerate(cell['source']):
            if 'confidence_threshold: 0.9' in line:
                cell['source'][i] = line.replace('confidence_threshold: 0.9', 'confidence_threshold: 0.266 (80th percentile)')
                print(f"  ✓ Updated markdown reference")

with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=1)

print("\n✓ All confidence_threshold values updated to 0.266 (80th percentile)")
