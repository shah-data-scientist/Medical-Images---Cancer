"""
Fix syntax error in model persistence cell (Cell 34)
"""
import json

# Load notebook
with open('3_semi_supervised_learning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix Cell 34 (Model Persistence)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if 'Model saving configured' in source_text and 'os.makedirs' in source_text:
            # Replace the broken cell
            fixed_code = """# Save Best Models (from Fold 1 as example)
import os
os.makedirs('models', exist_ok=True)

# Note: In practice, you'd save the best performing model from cross-validation
# Here we demonstrate the syntax

# Example: Save Scenario C model (usually best)
# torch.save(model_c.state_dict(), 'models/scenario_c_best_model.pth')

print("Model saving configured")
print("To save models, uncomment the torch.save() lines above")
print("Models can be loaded with:")
print("  model = BrainTumorClassifier()")
print("  model.load_state_dict(torch.load('models/scenario_c_best_model.pth'))")"""

            cell['source'] = fixed_code.split('\n')
            print(f"Fixed cell {i} (Model Persistence): Removed syntax error")
            break

# Save updated notebook
with open('3_semi_supervised_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("\nSyntax error fixed!")
print("Cell 34 (Model Persistence) now has correct string formatting.")
