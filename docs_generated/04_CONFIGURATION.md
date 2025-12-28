# Configuration Reference

**Source:** Derived from code
**Last Verified:** 2025-12-28

---

## Configuration Files

### pyproject.toml

**Purpose:** Project metadata and dependencies

**Key Configuration:**
```toml
[project]
name = "brainscanai"
version = "0.1.0"
description = "Brain tumor detection using semi-supervised learning"
requires-python = ">=3.11,<3.13"
```

**Dependencies:** 16 packages (see 02_HOW_TO_RUN.md for full list)

---

### .gitignore

**Purpose:** Files excluded from version control

**Excluded Directories:**
- `__pycache__/`, `.venv/`, `env/`, `venv/`
- `.ipynb_checkpoints/`
- `data/`, `features/` (large data files)
- `mlflow.db`, `mlruns/`, `mlartifacts/`
- `.vscode/`, `.claude/`
- `old_scripts/`, `old_notebooks/`

**Excluded File Patterns:**
- `*.py[cod]`, `*$py.class`
- `*.backup`, `*.log`
- `*.npy`, `*.pth` (model checkpoints)
- `.DS_Store`, `Thumbs.db`

**Rationale:** Prevents committing large binaries, temp files, and environment-specific configs

---

## Hardcoded Configuration (In Notebooks)

### Global Constants

```python
# Random seed (all notebooks)
RANDOM_STATE = 42

# Image processing (1_feature_extraction.ipynb)
IMAGE_SIZE = (224, 224)  # ResNet50 input requirement
PCA_COMPONENTS = 50      # Dimensionality reduction target

# Clustering (2_unsupervised_analysis.ipynb)
N_CLUSTERS = 2           # Binary classification
CONFIDENCE_THRESHOLD = 0.7

# Data split (3_semi_supervised_learning.ipynb)
TRAIN_SIZE = 0.80        # 80% for training pool
TEST_SIZE = 0.20         # 20% for final test

# Cross-validation
N_FOLDS = 5
STRATIFIED = True

# Model architecture
INPUT_SIZE = 50          # PCA components
HIDDEN_SIZE = 64         # Hidden neurons
DROPOUT_RATE = 0.70      # Dropout probability

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05      # L2 regularization
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1
```

**Note:** These are **hardcoded** in notebooks, not configurable via files or environment variables.

---

## Environment Variables

**Currently Used:** None

**Potential Future Configuration:**
```bash
# Example (not currently implemented)
export BRAINSCAN_DATA_DIR=/path/to/data
export BRAINSCAN_FEATURES_DIR=/path/to/features
export BRAINSCAN_RANDOM_SEED=42
export BRAINSCAN_MODEL_CHECKPOINT=/path/to/model.pth
```

---

## MLflow Configuration

**Tracking URI:** `file://./mlruns/` (local filesystem)

**Auto-configured:**
- Experiment tracking enabled by default
- Metrics logged: F2, precision, recall, accuracy
- Parameters logged: EPOCHS, BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE

**Access MLflow UI:**
```bash
poetry run mlflow ui
# Open http://localhost:5000
```

---

## Directory Structure (Required)

```
Medical-Images---Cancer/
├── data/                  # Required, gitignored
│   ├── labelled/
│   │   ├── no/
│   │   └── yes/
│   └── unlabelled/
│       ├── no/
│       └── yes/
├── features/              # Generated, gitignored
├── mlruns/                # Generated, gitignored
├── .venv/                 # Generated, gitignored
└── docs_generated/        # Generated documentation
```

**Critical:** `data/` directory must exist with proper structure before running notebooks.

---

## Known Limitations

1. **No Config File Support:**
   - No `.yaml`, `.json`, or `.ini` config files
   - All configuration hardcoded in notebooks

2. **No Environment Variable Support:**
   - Cannot override defaults via environment

3. **No CLI Arguments:**
   - Standalone scripts don't accept command-line arguments

4. **No Runtime Configuration:**
   - Cannot change hyperparameters without editing code

---

**Source:** Derived from:
- `pyproject.toml`
- `.gitignore`
- Hardcoded constants in notebooks
- MLflow auto-configuration
