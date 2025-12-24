import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find first code cell (index 2)
cell = nb['cells'][2]

# Fix line 59 - add newline
if len(cell['source']) > 59 and 'torch.cuda.get_device_name(0)")' in cell['source'][59]:
    if not cell['source'][59].endswith('\n'):
        cell['source'][59] = cell['source'][59] + '\n'
        print("Added newline to line 59")

# Remove duplicate import lines (60-61 are duplicates of 62-63)
# Keep only unique imports
if len(cell['source']) > 63:
    # Remove the first occurrence of duplicates (lines 60-61)
    # Actually, looking at the output, we have:
    # 60: 'from statsmodels.stats.contingency_tables import mcnemar\n'
    # 61: 'from sklearn.calibration import calibration_curve\n'
    # 62: 'from statsmodels.stats.contingency_tables import mcnemar\n'
    # We need to check if there are duplicates and remove them

    # Let's add a blank line after line 59, then keep lines 60-61
    if 'from statsmodels.stats.contingency_tables import mcnemar' in cell['source'][60]:
        # Remove duplicates if they exist
        if len(cell['source']) > 62 and cell['source'][62] == cell['source'][60]:
            # Remove line 62
            del cell['source'][62]
            print("Removed duplicate mcnemar import")
        if len(cell['source']) > 62 and 'from sklearn.calibration import calibration_curve' in cell['source'][62]:
            # Check if this is a duplicate
            if cell['source'][61] == cell['source'][62]:
                del cell['source'][62]
                print("Removed duplicate calibration import")

# Save fixed notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Newline and duplicate fixes applied!")
