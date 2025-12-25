"""
Quick fix: Replace 'combined_metadata' with 'metadata_df' in Notebook 2 Cell 14
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path("2_unsupervised_analysis.ipynb")

print("=" * 60)
print("FIXING VARIABLE NAME IN CELL 14")
print("=" * 60)

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\n[OK] Loaded notebook: {len(nb['cells'])} cells")

# Find and fix the clustering cell (contains 'combined_metadata')
fixed = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'combined_metadata' in source and 'kmeans' in source.lower():
            # Replace combined_metadata with metadata_df
            new_source = source.replace('combined_metadata', 'metadata_df')
            # Convert back to list of lines
            cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
            print(f"[OK] Fixed cell {i} (id={cell.get('id', 'unknown')}): replaced 'combined_metadata' with 'metadata_df'")
            fixed = True
            break

if not fixed:
    print("[ERROR] Could not find or fix Cell 14")
    exit(1)

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Saved updated notebook")
print("\n" + "=" * 60)
print("FIX COMPLETE")
print("=" * 60)
print("\nCHANGE:")
print("  BEFORE: combined_metadata['split']")
print("  AFTER:  metadata_df['split']")
print("\n[OK] Ready to re-run Notebook 2!")
