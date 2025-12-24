# VS Code Setup Guide for BrainScanAI

> **Complete guide to using Poetry virtual environment in VS Code**

---

## ✅ Environment Already Configured!

I've already set up:
- ✅ Poetry virtual environment in `.venv/`
- ✅ VS Code settings in `.vscode/settings.json`
- ✅ Jupyter kernel: "Python 3.11.9 (BrainScanAI)"

---

## 🚀 Quick Start (3 Steps)

### Step 1: Open Project in VS Code

```bash
# Navigate to project directory
cd "c:\Users\shahu\Documents\OneDrive\OPEN CLASSROOMS\PROJET 7\Medical Images - Cancer"

# Open in VS Code
code .
```

### Step 2: Select Python Interpreter

1. Press **Ctrl+Shift+P** (or Cmd+Shift+P on Mac)
2. Type: `Python: Select Interpreter`
3. Choose: **`.venv\Scripts\python.exe`** (should show Python 3.11.9)

**OR** click the Python version in the bottom-left status bar.

### Step 3: Open & Run Notebook

1. Open `1_feature_extraction.ipynb`
2. Click **"Select Kernel"** (top-right corner)
3. Choose **"Python 3.11.9 (BrainScanAI)"**
4. Run cells with **Shift+Enter**

---

## 🔧 Troubleshooting

### Issue 1: Can't See .venv in Interpreter List

**Solution 1 - Reload Window:**
```
Ctrl+Shift+P → "Developer: Reload Window"
```

**Solution 2 - Manual Path Entry:**
1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Click `+ Enter interpreter path...`
3. Paste: `${workspaceFolder}/.venv/Scripts/python.exe`

**Solution 3 - Restart VS Code:**
- Close VS Code completely
- Reopen the project folder

### Issue 2: Kernel Not Found in Notebooks

**Check available kernels:**
```bash
poetry run jupyter kernelspec list
```

**Expected output:**
```
Available kernels:
  brainscanai-vscode    C:\Users\shahu\AppData\Roaming\jupyter\kernels\brainscanai-vscode
  python3               ...
```

**If missing, reinstall:**
```bash
poetry run python -m ipykernel install --user --name=brainscanai-vscode --display-name="Python 3.11.9 (BrainScanAI)"
```

### Issue 3: "Module not found" Errors

**Verify environment:**
```bash
# Check Python path
poetry run python -c "import sys; print(sys.executable)"

# Should show: ...\.venv\Scripts\python.exe

# Test imports
poetry run python -c "import torch; print(torch.__version__)"
```

**If packages missing:**
```bash
poetry install
```

### Issue 4: Kernel Keeps Dying/Restarting

**Possible causes:**
- Out of memory (reduce batch size in notebooks)
- Corrupted cache (clear Jupyter cache)
- Environment conflict

**Solutions:**

**Clear Jupyter cache:**
```bash
poetry run jupyter --paths
# Delete contents of runtime and data directories
```

**Restart kernel in notebook:**
- Click kernel name → "Restart"

**Rebuild environment:**
```bash
poetry env remove python
poetry install
poetry run python -m ipykernel install --user --name=brainscanai-vscode --display-name="Python 3.11.9 (BrainScanAI)"
```

---

## 📝 VS Code Extensions (Recommended)

### Essential
1. **Python** (Microsoft) - Already required
2. **Jupyter** (Microsoft) - For notebooks

### Helpful
3. **Pylance** (Microsoft) - Better IntelliSense
4. **autoDocstring** - Generate docstrings
5. **Better Comments** - Colorized comments

### Install Extensions:
```
Ctrl+Shift+X → Search → Install
```

---

## ⚙️ Configured Settings

I've already created `.vscode/settings.json` with:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "jupyter.jupyterServerType": "local",
    "files.autoSave": "afterDelay"
}
```

**To view/edit:**
- Open `.vscode/settings.json` in the project

---

## 🎯 Running Notebooks in VS Code

### Method 1: Interactive (Recommended)

1. Open notebook (e.g., `1_feature_extraction.ipynb`)
2. Select kernel: "Python 3.11.9 (BrainScanAI)"
3. Click in any cell
4. Press **Shift+Enter** to run cell
5. Use **Ctrl+Enter** to run without moving to next cell

### Method 2: Run All Cells

1. Open notebook
2. Click **"Run All"** at the top (▶▶ icon)
3. Monitor progress in Output panel

### Method 3: Command Line

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace "1_feature_extraction.ipynb"
```

---

## 🔍 Useful VS Code Shortcuts

### Notebook Shortcuts
| Action | Shortcut |
|--------|----------|
| Run cell | `Shift+Enter` |
| Run cell (stay) | `Ctrl+Enter` |
| Insert cell above | `A` (command mode) |
| Insert cell below | `B` (command mode) |
| Delete cell | `DD` (command mode) |
| Change to markdown | `M` (command mode) |
| Change to code | `Y` (command mode) |
| Command palette | `Ctrl+Shift+P` |

### General Shortcuts
| Action | Shortcut |
|--------|----------|
| Terminal | `` Ctrl+` `` |
| Command palette | `Ctrl+Shift+P` |
| File explorer | `Ctrl+Shift+E` |
| Search | `Ctrl+Shift+F` |
| Extensions | `Ctrl+Shift+X` |

---

## 🐍 Using Integrated Terminal

### Open Terminal with Poetry Environment

**Option 1 - Automatic (via VS Code):**
1. Press `` Ctrl+` `` (backtick) to open terminal
2. VS Code should activate `.venv` automatically
3. Verify: prompt shows `(.venv)`

**Option 2 - Manual Activation:**
```bash
# PowerShell
.\.venv\Scripts\Activate.ps1

# Command Prompt
.venv\Scripts\activate.bat

# Git Bash
source .venv/Scripts/activate
```

**Option 3 - Use Poetry Shell:**
```bash
poetry shell
```

### Run Commands in Terminal

```bash
# With activated venv
python gather_stats.py
jupyter lab

# OR with poetry run (no activation needed)
poetry run python gather_stats.py
poetry run jupyter lab
```

---

## 📊 Checking Environment Status

### Verify Python Interpreter

**Bottom-left status bar** should show:
```
🐍 Python 3.11.9 64-bit ('.venv': poetry)
```

If not, click it and select the correct interpreter.

### Verify in Notebook

Run this in a notebook cell:
```python
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

import torch
print(f"PyTorch: {torch.__version__}")
```

**Expected output:**
```
Python: 3.11.9 ...
Executable: ...\.venv\Scripts\python.exe
PyTorch: 2.9.1
```

---

## 🎨 Customizing VS Code for Data Science

### Recommended Settings

Open `.vscode/settings.json` and add:

```json
{
    // Existing settings...

    // Notebook settings
    "jupyter.askForKernelRestart": false,
    "jupyter.widgetScriptSources": ["jsdelivr.com", "unpkg.com"],
    "notebook.lineNumbers": "on",
    "notebook.cellToolbarLocation": {
        "default": "right",
        "jupyter-notebook": "left"
    },

    // Editor settings
    "editor.rulers": [80, 120],
    "editor.formatOnSave": true,
    "files.trimTrailingWhitespace": true,

    // Python linting
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false
}
```

---

## 🚀 Performance Tips

### For Faster Notebook Execution

1. **Use Variable Inspector:**
   - Right-click in notebook → "Show Variable Inspector"
   - Monitor memory usage

2. **Clear Output:**
   - Right-click in notebook → "Clear All Outputs"
   - Reduces file size

3. **Restart Kernel:**
   - Click kernel name → "Restart"
   - Frees memory

### For Better IntelliSense

1. Ensure **Pylance** extension is installed
2. Set in settings:
   ```json
   "python.analysis.typeCheckingMode": "basic"
   ```

---

## 📦 Managing Dependencies

### Adding New Packages

**Via Terminal:**
```bash
poetry add package-name
```

**Via VS Code:**
1. Open terminal (`` Ctrl+` ``)
2. Run: `poetry add package-name`
3. Restart kernel in notebooks

### Updating Packages

```bash
# Update all
poetry update

# Update specific package
poetry update package-name
```

### Removing Packages

```bash
poetry remove package-name
```

---

## 🔄 Git Integration (Bonus)

### Recommended .gitignore

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Virtual environments
.venv/
venv/
ENV/

# VS Code
.vscode/
!.vscode/settings.json

# Data
features/
*.npy
*.pth
*.pkl

# OS
.DS_Store
Thumbs.db
```

### Initialize Git

```bash
git init
git add .
git commit -m "Initial commit: BrainScanAI project setup"
```

---

## ✅ Verification Checklist

Before running notebooks, verify:

- [ ] VS Code shows correct Python interpreter (`.venv\Scripts\python.exe`)
- [ ] Terminal shows `(.venv)` when activated
- [ ] Notebook kernel is "Python 3.11.9 (BrainScanAI)"
- [ ] Can import torch successfully
- [ ] No "module not found" errors

**Test Command:**
```bash
poetry run python -c "import torch, torchvision, numpy, pandas; print('All imports successful!')"
```

---

## 🆘 Still Having Issues?

### Check These:

1. **Restart VS Code completely** (not just reload window)
2. **Close all notebooks** before changing interpreter
3. **Verify Poetry environment exists**:
   ```bash
   poetry env info
   ```
4. **Reinstall kernel**:
   ```bash
   poetry run python -m ipykernel install --user --name=brainscanai-vscode --display-name="Python 3.11.9 (BrainScanAI)"
   ```

### Get System Info:

```bash
poetry env info
poetry show
python --version
code --version
```

---

## 📚 Additional Resources

- **VS Code Python Docs**: https://code.visualstudio.com/docs/python/python-tutorial
- **Jupyter in VS Code**: https://code.visualstudio.com/docs/datascience/jupyter-notebooks
- **Poetry Docs**: https://python-poetry.org/docs/

---

## ✨ Quick Reference Card

```
Open project:           code .
Select interpreter:     Ctrl+Shift+P → Python: Select Interpreter
Open terminal:          Ctrl+`
Run cell:              Shift+Enter
Command palette:        Ctrl+Shift+P
Restart kernel:        Click kernel name → Restart

Poetry commands:
poetry shell           # Activate environment
poetry add <pkg>       # Add package
poetry install         # Install dependencies
poetry run python      # Run Python script
```

---

**Last Updated**: 2025-12-24
**Python**: 3.11.9
**Poetry**: 2.2.1
**Environment**: `.venv/`

**You're ready to go! Open VS Code and start coding! 🚀**
