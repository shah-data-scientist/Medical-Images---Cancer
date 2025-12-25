"""
Fix Cell 25: Remove CONFIDENCE_THRESHOLD usage (undefined variable)
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path("2_unsupervised_analysis.ipynb")

print("=" * 60)
print("FIXING CONFIDENCE_THRESHOLD IN CELL 25")
print("=" * 60)

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\n[OK] Loaded notebook: {len(nb['cells'])} cells")

# Find Cell 25 (contains CONFIDENCE_THRESHOLD)
fixed = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'CONFIDENCE_THRESHOLD' in source and 'high_conf_labeled_mask' in source:
            # Replace the problematic line
            new_source = source.replace(
                "high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= CONFIDENCE_THRESHOLD",
                "# Calculate 80th percentile threshold from all labeled data\n"
                "labeled_threshold = np.percentile(strong_labeled_df['confidence_score'], 80)\n"
                "high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= labeled_threshold"
            )

            # Convert back to list of lines
            cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
            print(f"[OK] Fixed cell {i}: replaced CONFIDENCE_THRESHOLD with dynamic calculation")
            fixed = True
            break

if not fixed:
    print("[ERROR] Could not find or fix the cell with CONFIDENCE_THRESHOLD")
    exit(1)

# Save
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Saved updated notebook")
print("\n" + "=" * 60)
print("FIX COMPLETE")
print("=" * 60)
print("\nCHANGE:")
print("  BEFORE: high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= CONFIDENCE_THRESHOLD")
print("  AFTER:  labeled_threshold = np.percentile(strong_labeled_df['confidence_score'], 80)")
print("          high_conf_labeled_mask = strong_labeled_df['confidence_score'] >= labeled_threshold")
print("\n[OK] Ready to re-run Notebook 2!")
