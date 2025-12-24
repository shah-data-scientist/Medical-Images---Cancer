import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the threshold
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        for j, line in enumerate(cell['source']):
            if 'CONFIDENCE_THRESHOLD = 0.9' in line:
                cell['source'][j] = 'CONFIDENCE_THRESHOLD = 0.266  # 80th percentile (retains ~300 images)\n'
                print(f"Updated cell {i}, line {j}: threshold changed to 0.266")

# Save updated notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Threshold updated successfully!")
