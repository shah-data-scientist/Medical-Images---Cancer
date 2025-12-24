import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 30 (the one before "## 9. Export Final Data for Notebook 3")
cell_30 = nb['cells'][30]

print(f"Cell type: {cell_30['cell_type']}")
print(f"Number of lines: {len(cell_30['source'])}")
print("\nFirst 20 lines:")
for i, line in enumerate(cell_30['source'][:20]):
    print(f"{i}: {line[:100] if len(line) > 100 else line}", end='')
