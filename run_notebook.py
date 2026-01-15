
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import sys

if len(sys.argv) < 2:
    print("Usage: python run_notebook.py <notebook_path>")
    sys.exit(1)

notebook_path = sys.argv[1]

print(f"Reading notebook: {notebook_path}")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

class VerboseExecutePreprocessor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f"Executing cell {cell_index + 1}/{len(nb.cells)}...", flush=True)
            try:
                result = super().preprocess_cell(cell, resources, cell_index, **kwargs)
                print(f"Cell {cell_index + 1} Done", flush=True)
                return result
            except CellExecutionError as e:
                print(f"\nError in cell {cell_index + 1}: {e}", flush=True)
                raise
        else:
            return super().preprocess_cell(cell, resources, cell_index, **kwargs)

ep = VerboseExecutePreprocessor(timeout=3600, kernel_name='python3')  # Increased timeout to 1 hour

print("Starting execution...")
try:
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
    print("Execution complete.")
except Exception:
    print("\nStopping due to error.")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    sys.exit(1)

print(f"Saving executed notebook to: {notebook_path}")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Process finished successfully.")
